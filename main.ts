// free-model-aggregator-01: a quality-ranked free-model aggregator.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages), three upstream wires
// (OpenAI, Anthropic, Gemini) over ten simulated free-tier providers. It reads the brief, probes each
// provider's catalog, builds one catalog, registers it, serves the client's twenty requests, polls
// GET /work and exits 0 when done. `node main.ts --list` prints the catalog JSON and exits 0.
//
// Routing: every free model meeting a request's requirements is ranked by the quality table's
// rating_milli, with a per-call load penalty so calls spread over providers under their hidden
// per-trial budgets instead of hammering the single best one, and a penalty for a provider that just
// refused. A provider that refuses twice in a row is treated as dry and not used again this trial.
// Failover: a 402, 429 or 5xx (or an unreachable or unparseable upstream) retries the request on the
// next-best free model that meets it, on a different provider, counting the failed call.
// Wires: every upstream answer (JSON or SSE) is parsed into one canonical answer and rendered back in
// the client's wire, streamed or not as asked, with the request carried untouched (system, user text
// and images, tools, stop sequences, max_tokens, JSON mode) and usage and stop reasons mapped.
// A JSON-mode request never goes to an Anthropic-wire provider: that wire cannot echo response_format.
// Everything is a pure function of the brief and the providers' answers: no clock, no randomness, so a
// replay makes the same routing decisions and the same bytes.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicPassthrough,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropicRequest,
  openAIToGemini,
  openAIPassthrough,
  parseAnthropicAnswer,
  parseAnthropicStream,
  parseGeminiAnswer,
  parseGeminiStream,
  parseOpenAIAnswer,
  parseOpenAIStream,
  renderAnthropic,
  renderOpenAI,
  type CanonicalAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream call may take, in milliseconds, before it counts as a refusal. */
const UPSTREAM_TIMEOUT_MS = 20000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** One upstream call's rating penalty per call already made to that provider: spreads the load. */
const LOAD_PENALTY_MILLI = 8;
/** A provider's rating penalty per consecutive refusal: deprioritises it without killing it. */
const FAIL_PENALTY_MILLI = 10;
/** Consecutive refusals after which a provider is treated as dry for the rest of the trial. */
const DEAD_AFTER_FAILURES = 2;
/** The most upstream attempts for one client request (ten providers, plus a retry or two). */
const MAX_ATTEMPTS = 12;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

/** Per-provider trial state: calls made, consecutive refusals, and whether it is treated as dry. */
interface ProviderState {
  calls: number;
  failStreak: number;
  dead: boolean;
}
const providerState = new Map<string, ProviderState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  for (const probe of probes) providerState.set(probe.provider.id, { calls: 0, failStreak: 0, dead: false });
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  score: number;
}

/** Whether this client request asks for JSON mode (OpenAI only: Anthropic carries none). */
function wantsJson(api: 'openai' | 'anthropic', body: Json): boolean {
  if (api !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/**
 * Every free (provider, model) meeting the requirements, ranked by rating with the load and refusal
 * penalties, best first. A JSON-mode request excludes Anthropic-wire providers (that wire cannot echo
 * response_format); dry providers are excluded. `exclude` holds providers already tried for this
 * request; when every candidate is excluded the caller retries with `exclude` empty.
 */
function candidatesFor(requires: Requirement, api: 'openai' | 'anthropic', body: Json, exclude: ReadonlySet<string>): Candidate[] {
  const jsonMode = wantsJson(api, body);
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const state = providerState.get(probe.provider.id);
    if (state === undefined || state.dead) continue;
    if (exclude.has(probe.provider.id)) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      out.push({
        probe,
        model,
        rating,
        score: rating - LOAD_PENALTY_MILLI * state.calls - FAIL_PENALTY_MILLI * state.failStreak,
      });
    }
  }
  out.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(api: 'openai' | 'anthropic', body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  if (entry !== undefined && entry.api === api && entry.stream === (body.stream === true)) {
    manifestIndex += 1;
    return entry.requires;
  }
  // The manifest is the declared source; this fallback is a guess for a brief that holds none, or a
  // row whose api/stream does not match the request that arrived. Do not consume the row on a
  // mismatch: the manifest's next request is still the one at this index, and consuming the row
  // would desync every later request from its own requirements for the rest of the trial.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream URL, headers and body for one candidate and one client request. */
function upstreamRequest(candidate: Candidate, api: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; body: Json } {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    return {
      url: `${base}/v1/chat/completions`,
      headers,
      body: api === 'openai' ? openAIPassthrough(body, candidate.model.id) : anthropicToOpenAI(body, candidate.model.id),
    };
  }
  if (provider.wire === 'anthropic') {
    return {
      url: `${base}/v1/messages`,
      headers: { ...headers, 'anthropic-version': '2023-06-01' },
      body: api === 'anthropic' ? anthropicPassthrough(body, candidate.model.id) : openAIToAnthropicRequest(body, candidate.model.id),
    };
  }
  const gemini = api === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
  return { url: `${base}${gemini.path}`, headers, body: gemini.body };
}

/** One upstream response as a canonical answer, or null when it is not a well-formed answer. */
function parseUpstream(candidate: Candidate, stream: boolean, text: string): CanonicalAnswer | null {
  if (!stream) {
    let value: unknown;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRecord(value)) return null;
    if (candidate.probe.provider.wire === 'openai') return parseOpenAIAnswer(value);
    if (candidate.probe.provider.wire === 'anthropic') return parseAnthropicAnswer(value);
    return parseGeminiAnswer(value);
  }
  if (candidate.probe.provider.wire === 'openai') return parseOpenAIStream(text);
  if (candidate.probe.provider.wire === 'anthropic') return parseAnthropicStream(text);
  return parseGeminiStream(text);
}

function recordFailure(candidate: Candidate): void {
  const state = providerState.get(candidate.probe.provider.id);
  if (state === undefined) return;
  state.calls += 1;
  state.failStreak += 1;
  if (state.failStreak >= DEAD_AFTER_FAILURES) state.dead = true;
}

function recordSuccess(candidate: Candidate): void {
  const state = providerState.get(candidate.probe.provider.id);
  if (state === undefined) return;
  state.calls += 1;
  state.failStreak = 0;
}

/**
 * Serve one client request: rank the free candidates, call the best, and on a refusal retry the
 * next-best on another provider. Returns the rendered answer (and whether it was served), or null
 * when no provider could serve it.
 */
async function serve(body: Json, api: 'openai' | 'anthropic'): Promise<{ text: string; stream: boolean } | null> {
  const requires = requirementsFor(api, body);
  const stream = body.stream === true;
  const tried = new Set<string>();
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
    let candidates = candidatesFor(requires, api, body, tried);
    if (candidates.length === 0 && tried.size > 0) candidates = candidatesFor(requires, api, body, new Set());
    if (candidates.length === 0) return null;
    const candidate = candidates[0];
    const { url, headers, body: upstreamBody } = upstreamRequest(candidate, api, body);
    let upstream: Response;
    try {
      upstream = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(upstreamBody),
        signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
      });
    } catch (err) {
      log(`refusal: ${candidate.probe.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      recordFailure(candidate);
      tried.add(candidate.probe.provider.id);
      continue;
    }
    if (!upstream.ok) {
      log(`refusal: ${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${upstream.status}`);
      recordFailure(candidate);
      tried.add(candidate.probe.provider.id);
      continue;
    }
    const text = await upstream.text();
    const answer = parseUpstream(candidate, stream, text);
    if (answer === null || answer.text === '') {
      log(`refusal: ${candidate.probe.provider.id}/${candidate.model.id} gave no usable answer`);
      recordFailure(candidate);
      tried.add(candidate.probe.provider.id);
      continue;
    }
    recordSuccess(candidate);
    return { text: api === 'openai' ? renderOpenAI(answer, candidate.model.id, stream) : renderAnthropic(answer, candidate.model.id, stream), stream };
  }
  return null;
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const served = await serve(body, 'openai');
    if (served === null) return sendError(res, 'openai', 502, 'upstream_unreachable', 'no free model on the directory could serve this request');
    sendText(res, 200, served.stream ? 'text/event-stream' : 'application/json', served.text);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    const body = await readJson(req);
    if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
    const served = await serve(body, 'anthropic');
    if (served === null) return sendError(res, 'anthropic', 502, 'api_error', 'no free model on the directory could serve this request');
    sendText(res, 200, served.stream ? 'text/event-stream' : 'application/json', served.text);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
