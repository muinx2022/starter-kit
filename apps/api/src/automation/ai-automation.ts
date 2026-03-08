import fs from 'fs/promises';
import os from 'os';
import path from 'path';
import OpenAI from 'openai';
import type { Core } from '@strapi/strapi';

const SETTINGS_STORE = { type: 'core' as const, name: 'ai-automation', key: 'settings' };
const LOCK_STORE = { type: 'core' as const, name: 'ai-automation', key: 'locks' };
const LOCK_TTL_MS = 15 * 60 * 1000;
const DEFAULT_OPENAI_MODEL = process.env.OPENAI_AUTO_CONTENT_MODEL ?? 'gpt-4.1-mini';
const DEFAULT_ANTHROPIC_MODEL = process.env.ANTHROPIC_AUTO_CONTENT_MODEL ?? 'claude-haiku-4-5';
const DEFAULT_COMMENT_MODEL =
  process.env.OPENAI_AUTO_COMMENT_MODEL ?? process.env.OPENAI_AUTO_ENGAGE_MODEL ?? 'gpt-4.1-mini';
const DEFAULT_TIMEZONE = 'Asia/Ho_Chi_Minh';
const MAX_REMOTE_IMAGE_BYTES = parseInt(
  process.env.AI_AUTOMATION_MAX_IMAGE_BYTES ?? `${10 * 1024 * 1024}`,
  10
);
const FETCH_TIMEOUT_MS = parseInt(process.env.AI_AUTOMATION_FETCH_TIMEOUT_MS ?? '15000', 10);
const LEGACY_CONTENT_CATEGORY_ERROR = 'No valid published categories configured for content cron';

type UploadResult = { id: number; url?: string; alternativeText?: string | null; name?: string };
type MediaMode = 'body' | 'gallery';
type ImageProvider = 'auto' | 'google' | 'pexels';
type AiProvider = 'openai' | 'anthropic';
type ImageSearchProvider = Exclude<ImageProvider, 'auto'>;

export const PROVIDER_MODELS = {
  openai: ['gpt-4.1-mini', 'gpt-4.1', 'gpt-4o-mini', 'gpt-4o'],
  anthropic: ['claude-haiku-4-5', 'claude-sonnet-4-6', 'claude-opus-4-6'],
} as const;

export type AiAutomationJobStatus = {
  lastRunAt: string | null;
  lastSuccessAt: string | null;
  lastError: string | null;
};

export type AiContentCronSettings = AiAutomationJobStatus & {
  enabled: boolean;
  cron: string;
  postsPerRun: number;
  categoryDocumentIds: string[];
  scenarioPrompt: string;
  lastScenario: string | null;
  contentPrompt: string;
  imageProvider: ImageProvider;
  imageCountMin: number;
  imageCountMax: number;
  preferredMediaMode: MediaMode;
};

export type AiCommentCronSettings = AiAutomationJobStatus & {
  enabled: boolean;
  cron: string;
  commentsPerRun: number;
  allowReplies: boolean;
  commentPrompt: string;
  replyPrompt: string;
};

export type AiProviderSettings = {
  enabled: boolean;
  apiKey: string;
  models: string[];
};

export type AiImageSearchProviderSettings = {
  enabled: boolean;
  apiKey: string;
  searchEngineId?: string;
};

export type AiAutomationSettings = {
  timezone: string;
  content: AiContentCronSettings;
  comments: AiCommentCronSettings;
  providers: {
    openai: AiProviderSettings;
    anthropic: AiProviderSettings;
  };
  imageSearch: {
    google: AiImageSearchProviderSettings;
    pexels: AiImageSearchProviderSettings;
  };
};

export type AiAutomationRunResult = {
  job: 'content' | 'comments';
  createdPosts?: number;
  uploadedImages?: number;
  embeddedBodyImages?: number;
  galleryImages?: number;
  createdComments?: number;
  skipped: number;
  errors: string[];
};

export type AiContentAutomationTestResult = {
  job: 'content';
  provider: AiProvider;
  model: string;
  category: Pick<CategoryEntry, 'documentId' | 'name' | 'slug'>;
  images: Array<{
    provider: ImageSearchProvider;
    url: string;
    alt?: string;
  }>;
  preview: {
    selectedScenario: string;
    title: string;
    excerpt: string;
    bodyText: string;
    imageSearchQueries: string[];
    relatedTags: string[];
    mediaMode: MediaMode;
  };
  warnings: string[];
};

export type AiCommentAutomationTestResult = {
  job: 'comments';
  provider: AiProvider;
  model: string;
  post: Pick<PostEntry, 'documentId' | 'title'>;
  replyMode: 'top-level' | 'reply';
  preview: string;
};

type SeededUser = { id: number; documentId: string; username: string; email: string };
type CategoryEntry = { id: number; documentId: string; name: string; slug: string };
type PostEntry = {
  id: number;
  documentId: string;
  title: string;
  content: string;
  author?: { id: number; documentId?: string; username: string } | null;
};
type CommentEntry = { id: number; documentId: string; authorName: string; content: string };

type GeneratedContentPayload = {
  selectedScenario: string;
  title: string;
  excerpt: string;
  bodyText: string;
  imageSearchQueries: string[];
  relatedTags: string[];
  imageTheme?: string;
  mediaMode?: MediaMode;
};

type RemoteImageCandidate = {
  provider: Exclude<ImageProvider, 'auto'>;
  url: string;
  alt?: string;
};

type SelectedModel = {
  provider: AiProvider;
  model: string;
  apiKey: string;
};

const GENERIC_CONTENT_CATEGORY: CategoryEntry = {
  id: 0,
  documentId: 'generic-content',
  name: 'Du lich va trai nghiem',
  slug: 'du-lich-va-trai-nghiem',
};

const DEFAULT_CONTENT_SCENARIOS = [
  'An sang o mot quan via he la, khong phai quan quen',
  'Di xe do hoac xe lua duong dai',
  'Ngu homestay o tinh le, chu nha nguoi dia phuong',
  'Bat gap mot canh binh minh hoac hoang hon khong co tinh',
  'Ghe cho dia phuong buoi sang som (khong phai cho dem)',
  'An mot mon dac san vung mien lan dau',
  'Di bien trai mua hoac bien vang',
  'Leo nui hoac di trail ngan',
  'Mot buoi toi o nha cung gia dinh, khong di dau',
  'Ghe mot quan an nho o hem, khong co tren Google Maps',
  'Chuyen di tu phat, khong co ke hoach truoc',
  'Ngoi tren tau hoac pha ngam canh',
  'Kham pha mot khu pho cu hoac lang nghe',
  'An mot mon duong pho binh thuong nhung hom do ngon bat ngo',
  'Trai nghiem mua bat ngo khi dang di choi',
  'Nghi dem o khach san nho tinh le, view binh thuong nhung yen',
  'Di mot minh, khong co ban dong hanh',
  'Chuyen di cung ca gia dinh gom nguoi lon tuoi hoac tre nho',
  'Ghe mot quan cu sau nhieu nam khong quay lai',
  'Tinh co gap hoac tro chuyen voi nguoi la khi di choi',
].join('\n');

function getDefaultJobStatus(): AiAutomationJobStatus {
  return { lastRunAt: null, lastSuccessAt: null, lastError: null };
}

export function getDefaultAiAutomationSettings(): AiAutomationSettings {
  return {
    timezone: DEFAULT_TIMEZONE,
    content: {
      enabled: false,
      cron: '0 */6 * * *',
      postsPerRun: 1,
      categoryDocumentIds: [],
      scenarioPrompt: DEFAULT_CONTENT_SCENARIOS,
      lastScenario: null,
      contentPrompt:
        'Sau khi nhan duoc kich ban da chon, hay tao noi dung bai dang ngan bang tieng Viet dua tren kich ban do. ' +
        'Giong van tu nhien nhu nguoi that dang nhan tin hoac dang story, khong can chi chu, khong can van ve. ' +
        'Noi dung dai khoang 5 den 7 cau. Cam xuc that, don gian, kieu ke chuyen cho ban nghe. ' +
        'Tranh dung tu ngữ sao rong, my tu, kieu "buc tranh doi song", "khoanh khac dang tran trong", "hanh trinh y nghia". ' +
        'Thay bang ngon ngu binh thuong, de doc, doi thuong, neu hop canh co the hoi hoi xuong mot chut nhung van tu nhien. ' +
        'Co the ket bai bang 1 cau hoi hoac loi goi mo de tang tuong tac neu that su hop. Han che emoji, toi da 1-2 neu can. ' +
        'Hay tao excerpt ngan hon bodyText. Tao 3-6 relatedTags bang tieng Viet, ngan, sat noi dung. ' +
        'Tao 5-8 imageSearchQueries bang tieng Anh, uu tien them modifier nhu candid, phone photo, real life, amateur, everyday, casual, natural light, unfiltered, snapshot de tranh anh stock qua dep. ' +
        'Khong dung cac query kieu perfect travel photography hoac food photography.',
      imageProvider: 'auto',
      imageCountMin: 3,
      imageCountMax: 5,
      preferredMediaMode: 'body',
      ...getDefaultJobStatus(),
    },
    comments: {
      enabled: false,
      cron: '0 */6 * * *',
      commentsPerRun: 3,
      allowReplies: true,
      commentPrompt:
        'Hay viet 1 comment ngan, tu nhien, giong nguoi dung Viet Nam that. Binh dan, de doc, phu hop voi noi dung bai viet.',
      replyPrompt:
        'Hay viet 1 comment reply ngan, tu nhien, giong nguoi dung Viet Nam that. Van phong doi thuong, co lien quan truc tiep toi comment dang duoc reply.',
      ...getDefaultJobStatus(),
    },
    providers: {
      openai: {
        enabled: true,
        apiKey: '',
        models: [DEFAULT_OPENAI_MODEL, 'gpt-4o-mini'],
      },
      anthropic: {
        enabled: false,
        apiKey: '',
        models: [DEFAULT_ANTHROPIC_MODEL, 'claude-sonnet-4-6'],
      },
    },
    imageSearch: {
      google: {
        enabled: true,
        apiKey: '',
        searchEngineId: '',
      },
      pexels: {
        enabled: true,
        apiKey: '',
      },
    },
  };
}

function collapseWhitespace(value: string) {
  return value.replace(/\s+/g, ' ').trim();
}

function stripHtml(value: string) {
  return collapseWhitespace(String(value ?? '').replace(/<[^>]+>/g, ' '));
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function pickRandom<T>(items: T[]): T {
  return items[Math.floor(Math.random() * items.length)];
}

function shuffle<T>(items: T[]) {
  return [...items].sort(() => Math.random() - 0.5);
}

function parseScenarioPrompt(value: string) {
  return Array.from(
    new Set(
      String(value ?? '')
        .split(/\r?\n/)
        .map((item) => collapseWhitespace(item))
        .filter(Boolean)
    )
  );
}

function pickScenario(scenarios: string[], lastScenario?: string | null) {
  const normalizedLast = collapseWhitespace(String(lastScenario ?? ''));
  const candidates = scenarios.filter((item) => item !== normalizedLast);
  return pickRandom(candidates.length > 0 ? candidates : scenarios);
}

function escapeHtml(value: string) {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function slugify(value: string) {
  return String(value ?? '')
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 160);
}

function getMinuteKey(date: Date, timeZone: string) {
  const formatter = new Intl.DateTimeFormat('sv-SE', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
  return formatter.format(date).replace(' ', 'T');
}

function getZonedParts(date: Date, timeZone: string) {
  const formatter = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: 'numeric',
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric',
    weekday: 'short',
    hour12: false,
  });
  const parts = formatter.formatToParts(date);
  const get = (type: string) => parts.find((part) => part.type === type)?.value ?? '';
  const weekday = get('weekday');
  const weekdayMap: Record<string, number> = {
    Sun: 0,
    Mon: 1,
    Tue: 2,
    Wed: 3,
    Thu: 4,
    Fri: 5,
    Sat: 6,
  };
  return {
    minute: Number(get('minute')),
    hour: Number(get('hour')),
    dayOfMonth: Number(get('day')),
    month: Number(get('month')),
    dayOfWeek: weekdayMap[weekday] ?? 0,
  };
}

function parseNumber(value: string, min: number, max: number, aliases?: Record<string, number>) {
  const lowered = value.toLowerCase();
  if (aliases && lowered in aliases) {
    return aliases[lowered];
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) {
    throw new Error(`Invalid cron value "${value}"`);
  }
  return parsed;
}

function parseCronField(
  field: string,
  min: number,
  max: number,
  aliases?: Record<string, number>
): Set<number> {
  const trimmed = field.trim();
  if (!trimmed) {
    throw new Error('Cron field cannot be empty');
  }

  const values = new Set<number>();
  const segments = trimmed.split(',');
  for (const segment of segments) {
    const piece = segment.trim();
    if (!piece) {
      throw new Error(`Invalid cron segment "${field}"`);
    }

    const [base, stepValue] = piece.split('/');
    const step = stepValue === undefined ? 1 : Number(stepValue);
    if (!Number.isInteger(step) || step <= 0) {
      throw new Error(`Invalid cron step "${piece}"`);
    }

    let start = min;
    let end = max;

    if (base !== '*') {
      if (base.includes('-')) {
        const [from, to] = base.split('-');
        start = parseNumber(from, min, max, aliases);
        end = parseNumber(to, min, max, aliases);
      } else {
        start = parseNumber(base, min, max, aliases);
        end = start;
      }
    }

    if (start > end) {
      throw new Error(`Invalid cron range "${piece}"`);
    }

    for (let current = start; current <= end; current += step) {
      values.add(current === 7 && max === 7 ? 0 : current);
    }
  }

  return values;
}

export function validateCronExpression(expression: string) {
  const parts = String(expression ?? '')
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  if (parts.length !== 5) {
    throw new Error('Cron expression must have 5 fields');
  }

  parseCronField(parts[0], 0, 59);
  parseCronField(parts[1], 0, 23);
  parseCronField(parts[2], 1, 31);
  parseCronField(parts[3], 1, 12, {
    jan: 1,
    feb: 2,
    mar: 3,
    apr: 4,
    may: 5,
    jun: 6,
    jul: 7,
    aug: 8,
    sep: 9,
    oct: 10,
    nov: 11,
    dec: 12,
  });
  parseCronField(parts[4], 0, 7, {
    sun: 0,
    mon: 1,
    tue: 2,
    wed: 3,
    thu: 4,
    fri: 5,
    sat: 6,
  });
}

function cronMatches(expression: string, date: Date, timeZone: string) {
  const [minuteExpr, hourExpr, dayExpr, monthExpr, weekdayExpr] = expression.trim().split(/\s+/);
  const current = getZonedParts(date, timeZone);
  return (
    parseCronField(minuteExpr, 0, 59).has(current.minute) &&
    parseCronField(hourExpr, 0, 23).has(current.hour) &&
    parseCronField(dayExpr, 1, 31).has(current.dayOfMonth) &&
    parseCronField(monthExpr, 1, 12, {
      jan: 1,
      feb: 2,
      mar: 3,
      apr: 4,
      may: 5,
      jun: 6,
      jul: 7,
      aug: 8,
      sep: 9,
      oct: 10,
      nov: 11,
      dec: 12,
    }).has(current.month) &&
    parseCronField(weekdayExpr, 0, 7, {
      sun: 0,
      mon: 1,
      tue: 2,
      wed: 3,
      thu: 4,
      fri: 5,
      sat: 6,
    }).has(current.dayOfWeek)
  );
}

function normalizeSettings(input: Partial<AiAutomationSettings> | null | undefined): AiAutomationSettings {
  const defaults = getDefaultAiAutomationSettings();
  const merged: AiAutomationSettings = {
    timezone: String(input?.timezone ?? defaults.timezone).trim() || defaults.timezone,
    content: {
      ...defaults.content,
      ...(input?.content ?? {}),
    },
    comments: {
      ...defaults.comments,
      ...(input?.comments ?? {}),
    },
    providers: {
      openai: {
        ...defaults.providers.openai,
        ...(input?.providers?.openai ?? {}),
      },
      anthropic: {
        ...defaults.providers.anthropic,
        ...(input?.providers?.anthropic ?? {}),
      },
    },
    imageSearch: {
      google: {
        ...defaults.imageSearch.google,
        ...(input?.imageSearch?.google ?? {}),
      },
      pexels: {
        ...defaults.imageSearch.pexels,
        ...(input?.imageSearch?.pexels ?? {}),
      },
    },
  };

  merged.content.postsPerRun = clamp(Number(merged.content.postsPerRun) || defaults.content.postsPerRun, 1, 10);
  merged.content.categoryDocumentIds = Array.isArray(merged.content.categoryDocumentIds)
    ? merged.content.categoryDocumentIds.map((value) => String(value).trim()).filter(Boolean)
    : [];
  merged.content.scenarioPrompt =
    String(merged.content.scenarioPrompt ?? defaults.content.scenarioPrompt).trim() || defaults.content.scenarioPrompt;
  merged.content.lastScenario = merged.content.lastScenario ? String(merged.content.lastScenario).trim() : null;
  merged.content.contentPrompt = String(merged.content.contentPrompt ?? defaults.content.contentPrompt).trim();
  merged.content.imageProvider = ['auto', 'google', 'pexels'].includes(merged.content.imageProvider)
    ? merged.content.imageProvider
    : defaults.content.imageProvider;
  merged.content.imageCountMin = clamp(Number(merged.content.imageCountMin) || defaults.content.imageCountMin, 1, 10);
  merged.content.imageCountMax = clamp(Number(merged.content.imageCountMax) || defaults.content.imageCountMax, 1, 10);
  if (merged.content.imageCountMin > merged.content.imageCountMax) {
    const swappedMin = merged.content.imageCountMax;
    merged.content.imageCountMax = merged.content.imageCountMin;
    merged.content.imageCountMin = swappedMin;
  }
  merged.content.preferredMediaMode = merged.content.preferredMediaMode === 'gallery' ? 'gallery' : 'body';
  merged.content.enabled = Boolean(merged.content.enabled);
  merged.content.cron = String(merged.content.cron ?? defaults.content.cron).trim() || defaults.content.cron;
  merged.content.lastRunAt = merged.content.lastRunAt ? new Date(merged.content.lastRunAt).toISOString() : null;
  merged.content.lastSuccessAt = merged.content.lastSuccessAt ? new Date(merged.content.lastSuccessAt).toISOString() : null;
  merged.content.lastError = merged.content.lastError ? String(merged.content.lastError) : null;
  if (merged.content.lastError === LEGACY_CONTENT_CATEGORY_ERROR) {
    merged.content.lastError = null;
  }

  merged.comments.enabled = Boolean(merged.comments.enabled);
  merged.comments.cron = String(merged.comments.cron ?? defaults.comments.cron).trim() || defaults.comments.cron;
  merged.comments.commentsPerRun = clamp(
    Number(merged.comments.commentsPerRun) || defaults.comments.commentsPerRun,
    1,
    20
  );
  merged.comments.allowReplies = Boolean(merged.comments.allowReplies);
  merged.comments.commentPrompt =
    String(merged.comments.commentPrompt ?? defaults.comments.commentPrompt).trim() || defaults.comments.commentPrompt;
  merged.comments.replyPrompt =
    String(merged.comments.replyPrompt ?? defaults.comments.replyPrompt).trim() || defaults.comments.replyPrompt;
  merged.comments.lastRunAt = merged.comments.lastRunAt ? new Date(merged.comments.lastRunAt).toISOString() : null;
  merged.comments.lastSuccessAt = merged.comments.lastSuccessAt ? new Date(merged.comments.lastSuccessAt).toISOString() : null;
  merged.comments.lastError = merged.comments.lastError ? String(merged.comments.lastError) : null;

  merged.providers = {
    openai: {
      enabled: Boolean(input?.providers?.openai?.enabled ?? defaults.providers.openai.enabled),
      apiKey: String(input?.providers?.openai?.apiKey ?? defaults.providers.openai.apiKey),
      models: Array.isArray(input?.providers?.openai?.models)
        ? input!.providers!.openai!.models.map((value) => String(value).trim()).filter(Boolean)
        : defaults.providers.openai.models,
    },
    anthropic: {
      enabled: Boolean(input?.providers?.anthropic?.enabled ?? defaults.providers.anthropic.enabled),
      apiKey: String(input?.providers?.anthropic?.apiKey ?? defaults.providers.anthropic.apiKey),
      models: Array.isArray(input?.providers?.anthropic?.models)
        ? input!.providers!.anthropic!.models.map((value) => String(value).trim()).filter(Boolean)
        : defaults.providers.anthropic.models,
    },
  };

  merged.providers.openai.models = merged.providers.openai.models.filter((value) =>
    PROVIDER_MODELS.openai.includes(value as (typeof PROVIDER_MODELS.openai)[number])
  );
  merged.providers.anthropic.models = merged.providers.anthropic.models.filter((value) =>
    PROVIDER_MODELS.anthropic.includes(value as (typeof PROVIDER_MODELS.anthropic)[number])
  );

  merged.imageSearch = {
    google: {
      enabled: Boolean(input?.imageSearch?.google?.enabled ?? defaults.imageSearch.google.enabled),
      apiKey: String(input?.imageSearch?.google?.apiKey ?? defaults.imageSearch.google.apiKey).trim(),
      searchEngineId:
        String(input?.imageSearch?.google?.searchEngineId ?? defaults.imageSearch.google.searchEngineId).trim(),
    },
    pexels: {
      enabled: Boolean(input?.imageSearch?.pexels?.enabled ?? defaults.imageSearch.pexels.enabled),
      apiKey: String(input?.imageSearch?.pexels?.apiKey ?? defaults.imageSearch.pexels.apiKey).trim(),
    },
  };

  return merged;
}

export async function getAiAutomationSettings(strapi: Core.Strapi) {
  const store = strapi.store(SETTINGS_STORE);
  const current = (await store.get({})) as Partial<AiAutomationSettings> | null;
  return normalizeSettings(current);
}

async function saveAiAutomationSettings(strapi: Core.Strapi, settings: AiAutomationSettings) {
  const normalized = normalizeSettings(settings);
  const store = strapi.store(SETTINGS_STORE);
  await store.set({ value: normalized });
  return normalized;
}

export async function updateAiAutomationSettings(
  strapi: Core.Strapi,
  payload: Partial<AiAutomationSettings>
) {
  const current = await getAiAutomationSettings(strapi);
  const next = normalizeSettings({
    ...current,
    ...payload,
    content: {
      ...current.content,
      ...(payload.content ?? {}),
      lastRunAt: current.content.lastRunAt,
      lastSuccessAt: current.content.lastSuccessAt,
      lastError: current.content.lastError,
      lastScenario: current.content.lastScenario,
    },
    comments: {
      ...current.comments,
      ...(payload.comments ?? {}),
      lastRunAt: current.comments.lastRunAt,
      lastSuccessAt: current.comments.lastSuccessAt,
      lastError: current.comments.lastError,
    },
    providers: {
      openai: {
        ...current.providers.openai,
        ...(payload.providers?.openai ?? {}),
      },
      anthropic: {
        ...current.providers.anthropic,
        ...(payload.providers?.anthropic ?? {}),
      },
    },
    imageSearch: {
      google: {
        ...current.imageSearch.google,
        ...(payload.imageSearch?.google ?? {}),
      },
      pexels: {
        ...current.imageSearch.pexels,
        ...(payload.imageSearch?.pexels ?? {}),
      },
    },
  });

  validateCronExpression(next.content.cron);
  validateCronExpression(next.comments.cron);

  if (next.content.categoryDocumentIds.length > 0) {
    const publishedCategories = await fetchConfiguredCategories(strapi, next.content.categoryDocumentIds);
    if (publishedCategories.length === 0) {
      throw new Error('Selected content categories must include at least one published category');
    }
  }
  if (!next.content.contentPrompt) {
    throw new Error('Content prompt is required');
  }
  if (parseScenarioPrompt(next.content.scenarioPrompt).length === 0) {
    throw new Error('At least one content scenario is required');
  }
  if (!next.comments.commentPrompt) {
    throw new Error('Comment prompt is required');
  }
  if (!next.comments.replyPrompt) {
    throw new Error('Reply prompt is required');
  }
  if (next.providers.openai.enabled && next.providers.openai.models.length === 0) {
    throw new Error('OpenAI requires at least one selected model');
  }
  if (next.providers.anthropic.enabled && next.providers.anthropic.models.length === 0) {
    throw new Error('Anthropic requires at least one selected model');
  }

  return saveAiAutomationSettings(strapi, next);
}

async function fetchSeededUsers(strapi: Core.Strapi): Promise<SeededUser[]> {
  return (await strapi.db.query('plugin::users-permissions.user').findMany({
    where: { isSeeded: true, blocked: false, confirmed: true },
    limit: 200,
  })) as SeededUser[];
}

async function fetchPublishedPosts(strapi: Core.Strapi, limit: number) {
  return (await strapi.db.query('api::post.post').findMany({
    where: { publishedAt: { $notNull: true } },
    orderBy: { createdAt: 'asc' },
    limit,
  })) as PostEntry[];
}

async function fetchConfiguredCategories(strapi: Core.Strapi, documentIds: string[]) {
  if (documentIds.length === 0) {
    return [] as CategoryEntry[];
  }

  return (await strapi.documents('api::category.category').findMany({
    filters: { documentId: { $in: documentIds } },
    fields: ['id', 'documentId', 'name', 'slug'],
    status: 'published',
    pagination: { page: 1, pageSize: Math.max(documentIds.length, 20) },
  })) as CategoryEntry[];
}

async function fetchAnyPublishedCategory(strapi: Core.Strapi) {
  const categories = (await strapi.documents('api::category.category').findMany({
    fields: ['id', 'documentId', 'name', 'slug'],
    status: 'published',
    pagination: { page: 1, pageSize: 1 },
  })) as CategoryEntry[];
  return categories[0] ?? null;
}

async function fetchPublishedCategories(strapi: Core.Strapi, pageSize = 200) {
  return (await strapi.documents('api::category.category').findMany({
    fields: ['id', 'documentId', 'name', 'slug'],
    status: 'published',
    pagination: { page: 1, pageSize },
  })) as CategoryEntry[];
}

async function fetchContentCategories(strapi: Core.Strapi, categoryDocumentIds: string[]) {
  if (categoryDocumentIds.length > 0) {
    return fetchConfiguredCategories(strapi, categoryDocumentIds);
  }
  return fetchPublishedCategories(strapi);
}

async function findExistingTagDocumentId(strapi: Core.Strapi, name: string, slug: string) {
  const matches = (await strapi.entityService.findMany('api::tag.tag', {
    fields: ['documentId', 'name', 'slug'],
    filters: {
      $or: [{ name: { $eqi: name } }, { slug: { $eq: slug } }],
    },
    publicationState: 'preview',
    limit: 10,
  } as any)) as Array<{ documentId?: string }>;

  return String(matches?.[0]?.documentId ?? '').trim() || null;
}

async function ensureTagDocumentIds(strapi: Core.Strapi, rawTags: string[]) {
  const uniqueTags = Array.from(
    new Map(
      rawTags
        .map((value) => collapseWhitespace(String(value ?? '')))
        .filter(Boolean)
        .map((value) => [value.toLowerCase(), value])
    ).values()
  ).slice(0, 6);

  const documentIds: string[] = [];
  for (const tagName of uniqueTags) {
    const slug = slugify(tagName);
    if (!slug) {
      continue;
    }

    const existingDocumentId = await findExistingTagDocumentId(strapi, tagName, slug);
    if (existingDocumentId) {
      documentIds.push(existingDocumentId);
      continue;
    }

    const created = (await strapi.documents('api::tag.tag').create({
      data: {
        name: tagName,
        slug,
      },
      status: 'draft',
    })) as { documentId?: string } | null;

    const createdDocumentId = String(created?.documentId ?? '').trim();
    if (createdDocumentId) {
      documentIds.push(createdDocumentId);
    }
  }

  return documentIds;
}

function resolveProviderApiKey(settings: AiAutomationSettings, provider: AiProvider) {
  if (provider === 'openai') {
    return settings.providers.openai.apiKey.trim() || process.env.OPENAI_API_KEY || '';
  }
  return settings.providers.anthropic.apiKey.trim() || process.env.ANTHROPIC_API_KEY || '';
}

function getEnabledModels(settings: AiAutomationSettings): SelectedModel[] {
  const entries: SelectedModel[] = [];

  if (settings.providers.openai.enabled) {
    const apiKey = resolveProviderApiKey(settings, 'openai');
    if (apiKey) {
      for (const model of settings.providers.openai.models) {
        entries.push({
          provider: 'openai',
          model,
          apiKey,
        });
      }
    }
  }

  if (settings.providers.anthropic.enabled) {
    const apiKey = resolveProviderApiKey(settings, 'anthropic');
    if (apiKey) {
      for (const model of settings.providers.anthropic.models) {
        entries.push({
          provider: 'anthropic',
          model,
          apiKey,
        });
      }
    }
  }

  return entries;
}

function resolveImageSearchApiKey(settings: AiAutomationSettings, provider: ImageSearchProvider) {
  if (provider === 'google') {
    return settings.imageSearch.google.apiKey || process.env.GOOGLE_SEARCH_API_KEY || '';
  }
  return settings.imageSearch.pexels.apiKey || process.env.PEXELS_API_KEY || '';
}

function resolveGoogleSearchEngineId(settings: AiAutomationSettings) {
  return settings.imageSearch.google.searchEngineId || process.env.GOOGLE_SEARCH_ENGINE_ID || '';
}

function isImageSearchProviderEnabled(settings: AiAutomationSettings, provider: ImageSearchProvider) {
  return settings.imageSearch[provider].enabled;
}

function hasImageProviderCredentials(settings: AiAutomationSettings, provider: ImageProvider) {
  if (provider === 'google') {
    return Boolean(
      isImageSearchProviderEnabled(settings, 'google') &&
        resolveImageSearchApiKey(settings, 'google') &&
        resolveGoogleSearchEngineId(settings)
    );
  }
  if (provider === 'pexels') {
    return Boolean(isImageSearchProviderEnabled(settings, 'pexels') && resolveImageSearchApiKey(settings, 'pexels'));
  }
  return hasImageProviderCredentials(settings, 'google') || hasImageProviderCredentials(settings, 'pexels');
}

async function generateProviderText(
  selected: SelectedModel,
  input: {
    systemPrompt: string;
    userPrompt: string;
    maxTokens: number;
    jsonMode?: boolean;
  }
) {
  if (selected.provider === 'openai') {
    const openai = new OpenAI({ apiKey: selected.apiKey });
    const response = await openai.responses.create({
      model: selected.model,
      instructions: input.systemPrompt.trim(),
      input: [
        {
          role: 'user',
          content: [{ type: 'input_text', text: input.userPrompt }],
        },
      ],
      max_output_tokens: input.maxTokens,
      ...(input.jsonMode
        ? {
            text: {
              format: {
                type: 'json_object',
              },
            },
          }
        : {}),
    } as any);

    return response.output_text.trim();
  }

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': selected.apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: selected.model,
      max_tokens: input.maxTokens,
      system: input.systemPrompt.trim(),
      messages: [{ role: 'user', content: input.userPrompt }],
    }),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(`Anthropic request failed (${response.status}): ${errorText.slice(0, 200)}`);
  }

  const payload = (await response.json()) as {
    content?: Array<{ type?: string; text?: string }>;
  };
  const text = (payload.content ?? [])
    .filter((item) => item.type === 'text' && item.text)
    .map((item) => item.text)
    .join('\n');

  return text.trim();
}

export async function checkAiProviderConnection(input: {
  provider: AiProvider;
  apiKey: string;
  model?: string;
}) {
  const providerModels = PROVIDER_MODELS[input.provider];
  const model = input.model && providerModels.includes(input.model as never) ? input.model : providerModels[0];

  if (input.provider === 'openai') {
    const openai = new OpenAI({ apiKey: input.apiKey });
    await openai.models.retrieve(model);
    return {
      ok: true,
      provider: input.provider,
      model,
      message: `Connected to OpenAI with model ${model}`,
    };
  }

  const response = await fetch('https://api.anthropic.com/v1/models', {
    headers: {
      'x-api-key': input.apiKey,
      'anthropic-version': '2023-06-01',
    },
  });
  if (!response.ok) {
    const errorText = await response.text().catch(() => '');
    throw new Error(`Anthropic connection failed (${response.status}): ${errorText.slice(0, 200)}`);
  }

  return {
    ok: true,
    provider: input.provider,
    model,
    message: `Connected to Anthropic with model ${model}`,
  };
}

async function generateStructuredContent(
  selected: SelectedModel,
  prompt: string,
  category: CategoryEntry,
  scenario: string
): Promise<GeneratedContentPayload> {
  const raw = await generateProviderText(selected, {
    systemPrompt:
      'Ban la tac gia Viet Nam viet bai dang ngan cho he thong noi dung. ' +
      'Tra ve json hop le duy nhat, khong markdown, khong giai thich. ' +
      'json gom: title, excerpt, bodyText, imageSearchQueries, relatedTags, imageTheme, mediaMode. ' +
      'Title phai co y nghia ro, co cam xuc, tu nhien. ' +
      'bodyText chi 1 doan ngan, 5-7 cau ngan, de doc. excerpt ngan hon bodyText. ' +
      'imageSearchQueries la mang 5-8 cum tu tim anh bang tieng Anh, phu hop voi bai viet. ' +
      'relatedTags la mang 3-6 tag ngan bang tieng Viet, viet thuong, sat chu de bai viet. ' +
      'mediaMode uu tien "body".',
    userPrompt:
      `Category: ${category.name}\n` +
      `Selected scenario: ${scenario}\n` +
      `Content prompt: ${prompt}\n` +
      'Viet bai dang bang tieng Viet. Chu de gan voi du lich, trai nghiem, cam xuc, khoanh khac doi thuong. Output must be valid json.',
    maxTokens: 700,
    jsonMode: selected.provider === 'openai',
  });

  const jsonCandidate = raw.includes('{') ? raw.slice(raw.indexOf('{'), raw.lastIndexOf('}') + 1) : raw;
  let parsed: Partial<GeneratedContentPayload>;
  try {
    parsed = JSON.parse(jsonCandidate) as Partial<GeneratedContentPayload>;
  } catch (error) {
    throw new Error(
      `AI returned invalid JSON: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  const title = collapseWhitespace(String(parsed.title ?? ''));
  const excerpt = collapseWhitespace(String(parsed.excerpt ?? ''));
  const bodyText = collapseWhitespace(String(parsed.bodyText ?? ''));
  const queries = Array.isArray(parsed.imageSearchQueries)
    ? parsed.imageSearchQueries.map((item) => collapseWhitespace(String(item))).filter(Boolean)
    : [];
  const relatedTags = Array.isArray(parsed.relatedTags)
    ? parsed.relatedTags
        .map((item) => collapseWhitespace(String(item)).toLowerCase())
        .filter(Boolean)
        .slice(0, 6)
    : [];

  if (!title || !bodyText) {
    throw new Error('AI returned incomplete content payload');
  }

  return {
    selectedScenario: scenario,
    title,
    excerpt: excerpt || bodyText.slice(0, 180),
    bodyText,
    imageSearchQueries: queries.length > 0 ? queries.slice(0, 8) : [category.name, title],
    relatedTags,
    imageTheme: collapseWhitespace(String(parsed.imageTheme ?? category.name)),
    mediaMode: parsed.mediaMode === 'gallery' ? 'gallery' : 'body',
  };
}

async function generateStructuredContentWithFallback(
  modelPool: SelectedModel[],
  prompt: string,
  category: CategoryEntry,
  scenario: string
) {
  const errors: string[] = [];
  const attempts = Math.min(3, Math.max(1, modelPool.length > 0 ? 3 : 1));
  const shuffled = shuffle(modelPool);

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const selectedModel = shuffled[attempt % shuffled.length];
    try {
      const generated = await generateStructuredContent(selectedModel, prompt, category, scenario);
      return { selectedModel, generated, errors };
    } catch (error) {
      errors.push(
        `attempt ${attempt + 1}/${attempts} ${selectedModel.provider}/${selectedModel.model}: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  throw new Error(
    errors[errors.length - 1] ?? `Unable to generate content after ${attempts} attempts`
  );
}

async function searchGoogleImages(
  settings: AiAutomationSettings,
  query: string,
  count: number
): Promise<RemoteImageCandidate[]> {
  const apiKey = resolveImageSearchApiKey(settings, 'google');
  const searchEngineId = resolveGoogleSearchEngineId(settings);
  if (!isImageSearchProviderEnabled(settings, 'google') || !apiKey || !searchEngineId) {
    return [];
  }

  const url =
    `https://www.googleapis.com/customsearch/v1?key=${encodeURIComponent(apiKey)}` +
    `&cx=${encodeURIComponent(searchEngineId)}` +
    `&q=${encodeURIComponent(query)}` +
    `&searchType=image&num=${Math.min(Math.max(count, 1), 10)}&safe=active`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Google image search failed (${response.status})`);
  }
  const payload = (await response.json()) as {
    items?: Array<{ link?: string; title?: string; snippet?: string }>;
  };
  return (payload.items ?? [])
    .map((item) => ({
      provider: 'google' as const,
      url: item.link ?? '',
      alt: item.title ?? item.snippet ?? undefined,
    }))
    .filter((item) => Boolean(item.url));
}

async function searchPexelsImages(
  settings: AiAutomationSettings,
  query: string,
  count: number
): Promise<RemoteImageCandidate[]> {
  const apiKey = resolveImageSearchApiKey(settings, 'pexels');
  if (!isImageSearchProviderEnabled(settings, 'pexels') || !apiKey) {
    return [];
  }

  const url = `https://api.pexels.com/v1/search?query=${encodeURIComponent(query)}&orientation=landscape&per_page=${Math.max(
    count * 3,
    10
  )}`;
  const response = await fetch(url, {
    headers: { Authorization: apiKey },
  });
  if (!response.ok) {
    throw new Error(`Pexels search failed (${response.status})`);
  }
  const payload = (await response.json()) as {
    photos?: Array<{ alt?: string; src?: { large2x?: string; large?: string; original?: string } }>;
  };
  return (payload.photos ?? [])
    .map((photo) => ({
      provider: 'pexels' as const,
      url: photo.src?.large2x ?? photo.src?.large ?? photo.src?.original ?? '',
      alt: photo.alt,
    }))
    .filter((photo) => Boolean(photo.url));
}

async function searchRemoteImages(
  settings: AiAutomationSettings,
  query: string,
  count: number,
  provider: ImageProvider
): Promise<RemoteImageCandidate[]> {
  if (provider === 'google') {
    return searchGoogleImages(settings, query, count);
  }
  if (provider === 'pexels') {
    return searchPexelsImages(settings, query, count);
  }

  const google = await searchGoogleImages(settings, query, count).catch(() => []);
  if (google.length > 0) {
    return google;
  }
  return searchPexelsImages(settings, query, count).catch(() => []);
}

async function downloadRemoteImage(candidate: RemoteImageCandidate, index: number, slugBase: string) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    const response = await fetch(candidate.url, {
      signal: controller.signal,
      headers: { 'User-Agent': 'trekky-strapi-ai-automation/1.0' },
    });
    if (!response.ok) {
      throw new Error(`Image download failed (${response.status})`);
    }

    const contentType = String(response.headers.get('content-type') ?? '').toLowerCase();
    if (!contentType.startsWith('image/')) {
      throw new Error(`Unsupported image content-type "${contentType}"`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    if (buffer.byteLength === 0) {
      throw new Error('Downloaded image is empty');
    }
    if (buffer.byteLength > MAX_REMOTE_IMAGE_BYTES) {
      throw new Error('Downloaded image exceeds size limit');
    }

    const ext = contentType.includes('png')
      ? '.png'
      : contentType.includes('webp')
        ? '.webp'
        : contentType.includes('gif')
          ? '.gif'
          : '.jpg';

    const filename = `${slugBase}-${index + 1}${ext}`;
    const filepath = path.join(os.tmpdir(), filename);
    await fs.writeFile(filepath, buffer);

    return {
      filepath,
      originalFilename: filename,
      mimetype: contentType,
      size: buffer.byteLength,
      alt: candidate.alt,
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function uploadLocalImages(
  strapi: Core.Strapi,
  downloads: Array<{ filepath: string; originalFilename: string; mimetype: string; size: number; alt?: string }>
) {
  if (downloads.length === 0) {
    return [] as UploadResult[];
  }

  try {
    const uploadService = strapi.plugin('upload').service('upload') as {
      upload(args: {
        data: Record<string, unknown>;
        files: Array<{
          filepath: string;
          originalFilename: string;
          mimetype: string;
          size: number;
        }>;
      }): Promise<UploadResult[]>;
    };

    return await uploadService.upload({
      data: {
        fileInfo: downloads.map((file) => ({
          name: file.originalFilename,
          alternativeText: file.alt ?? null,
        })),
      },
      files: downloads.map((file) => ({
        filepath: file.filepath,
        originalFilename: file.originalFilename,
        mimetype: file.mimetype,
        size: file.size,
      })),
    });
  } finally {
    await Promise.all(downloads.map((file) => fs.unlink(file.filepath).catch(() => undefined)));
  }
}

function buildBodyHtml(bodyText: string, images: UploadResult[]) {
  const intro = `<p>${escapeHtml(bodyText)}</p>`;
  const list = images
    .map((image) => {
      const alt = escapeHtml(image.alternativeText || image.name || 'Post image');
      const src = image.url ?? '';
      return `<p><img src="${src}" alt="${alt}" /></p>`;
    })
    .join('');
  return `${intro}${list}`;
}

async function createDraftPost(
  strapi: Core.Strapi,
  payload: {
    title: string;
    excerpt: string;
    content: string;
    authorId: number;
    categoryDocumentIds: string[];
    tagDocumentIds: string[];
    imageIds: number[];
    aiSource: {
      provider: AiProvider;
      model: string;
      generatedAt: string;
    };
  }
) {
  return strapi.documents('api::post.post').create({
    data: {
      title: payload.title,
      slug: slugify(payload.title),
      excerpt: payload.excerpt,
      content: payload.content,
      author: payload.authorId,
      categories: payload.categoryDocumentIds,
      tags: payload.tagDocumentIds,
      images: payload.imageIds,
      aiSource: payload.aiSource,
    },
    status: 'draft',
  });
}

async function persistJobStatus(
  strapi: Core.Strapi,
  job: 'content' | 'comments',
  patch: Partial<AiAutomationJobStatus> & { lastScenario?: string | null }
) {
  const current = await getAiAutomationSettings(strapi);
  const currentJob = current[job];
  await saveAiAutomationSettings(strapi, {
    ...current,
    [job]: {
      ...currentJob,
      ...patch,
    },
  } as AiAutomationSettings);
}

async function withJobLock<T>(
  strapi: Core.Strapi,
  job: 'content' | 'comments',
  work: () => Promise<T>
) {
  const store = strapi.store(LOCK_STORE);
  const current = ((await store.get({})) ?? {}) as Record<string, { lockedAt?: string }>;
  const lockedAt = current[job]?.lockedAt ? new Date(current[job].lockedAt).getTime() : 0;
  if (lockedAt && Date.now() - lockedAt < LOCK_TTL_MS) {
    throw new Error(`${job} job is already running`);
  }

  await store.set({
    value: {
      ...current,
      [job]: { lockedAt: new Date().toISOString() },
    },
  });

  try {
    return await work();
  } finally {
    const latest = ((await store.get({})) ?? {}) as Record<string, { lockedAt?: string }>;
    delete latest[job];
    await store.set({ value: latest });
  }
}

export async function runContentAutomation(strapi: Core.Strapi): Promise<AiAutomationRunResult> {
  return withJobLock(strapi, 'content', async () => {
    const settings = await getAiAutomationSettings(strapi);
    const result: AiAutomationRunResult = {
      job: 'content',
      createdPosts: 0,
      uploadedImages: 0,
      embeddedBodyImages: 0,
      galleryImages: 0,
      skipped: 0,
      errors: [],
    };

    const nowIso = new Date().toISOString();
    await persistJobStatus(strapi, 'content', { lastRunAt: nowIso, lastError: null });

    try {
      const [users, categories] = await Promise.all([
        fetchSeededUsers(strapi),
        fetchContentCategories(strapi, settings.content.categoryDocumentIds),
      ]);
      const modelPool = getEnabledModels(settings);

      if (users.length === 0) {
        throw new Error('No seeded users available for AI content');
      }
      if (modelPool.length === 0) {
        throw new Error('No enabled AI provider/model configured for content generation');
      }

      for (let index = 0; index < settings.content.postsPerRun; index += 1) {
        try {
          const author = pickRandom(users);
          const category = categories.length > 0 ? pickRandom(categories) : GENERIC_CONTENT_CATEGORY;
          const scenarios = parseScenarioPrompt(settings.content.scenarioPrompt);
          const scenario = pickScenario(scenarios, settings.content.lastScenario);
          const {
            selectedModel,
            generated,
            errors: generationErrors,
          } = await generateStructuredContentWithFallback(
            modelPool,
            settings.content.contentPrompt,
            category,
            scenario
          );
          result.errors.push(...generationErrors);
          const slugBase = slugify(generated.title) || `ai-post-${Date.now()}`;
          let uploaded: UploadResult[] = [];

          if (hasImageProviderCredentials(settings, settings.content.imageProvider)) {
            try {
              const desiredImageCount = clamp(
                Math.floor(Math.random() * (settings.content.imageCountMax - settings.content.imageCountMin + 1)) +
                  settings.content.imageCountMin,
                settings.content.imageCountMin,
                settings.content.imageCountMax
              );

              const remoteImages: RemoteImageCandidate[] = [];
              for (const query of generated.imageSearchQueries) {
                const batch = await searchRemoteImages(settings, query, desiredImageCount, settings.content.imageProvider);
                for (const item of batch) {
                  if (!remoteImages.some((existing) => existing.url === item.url)) {
                    remoteImages.push(item);
                  }
                  if (remoteImages.length >= desiredImageCount) {
                    break;
                  }
                }
                if (remoteImages.length >= desiredImageCount) {
                  break;
                }
              }

              if (remoteImages.length === 0) {
                result.errors.push(`[content:${generated.title}] No remote images found, created text-only post`);
              } else {
                const selectedImages = remoteImages.sort(() => Math.random() - 0.5).slice(0, desiredImageCount);
                const downloads = [];

                for (let imageIndex = 0; imageIndex < selectedImages.length; imageIndex += 1) {
                  try {
                    downloads.push(await downloadRemoteImage(selectedImages[imageIndex], imageIndex, slugBase));
                  } catch (error) {
                    result.errors.push(
                      `[content:${generated.title}] image ${imageIndex + 1}: ${error instanceof Error ? error.message : String(error)}`
                    );
                  }
                }

                if (downloads.length === 0) {
                  result.errors.push(`[content:${generated.title}] Unable to download images, created text-only post`);
                } else {
                  uploaded = await uploadLocalImages(strapi, downloads);
                }
              }
            } catch (error) {
              result.errors.push(
                `[content:${generated.title}] image pipeline failed: ${error instanceof Error ? error.message : String(error)}`
              );
            }
          } else {
            result.errors.push(
              `[content:${generated.title}] Missing image search provider credentials, created text-only post`
            );
          }

          const uploadedIds = uploaded.map((item) => item.id).filter((id) => Number.isFinite(id));
          result.uploadedImages = (result.uploadedImages ?? 0) + uploadedIds.length;
          const tagDocumentIds = await ensureTagDocumentIds(strapi, generated.relatedTags);

          const preferredMode = generated.mediaMode === 'gallery' ? 'gallery' : settings.content.preferredMediaMode;
          const useBodyMode = preferredMode === 'body' && uploaded.length > 0;
          const bodyContent = useBodyMode
            ? buildBodyHtml(generated.bodyText, uploaded)
            : `<p>${escapeHtml(generated.bodyText)}</p>`;

          await createDraftPost(strapi, {
            title: generated.title,
            excerpt: generated.excerpt,
            content: bodyContent,
            authorId: author.id,
            categoryDocumentIds: category.id ? [category.documentId] : [],
            tagDocumentIds,
            imageIds: useBodyMode ? [] : uploadedIds,
            aiSource: {
              provider: selectedModel.provider,
              model: selectedModel.model,
              generatedAt: new Date().toISOString(),
            },
          });

          result.createdPosts = (result.createdPosts ?? 0) + 1;
          if (useBodyMode) {
            result.embeddedBodyImages = (result.embeddedBodyImages ?? 0) + uploaded.length;
          } else {
            result.galleryImages = (result.galleryImages ?? 0) + uploaded.length;
          }
          settings.content.lastScenario = scenario;
          await persistJobStatus(strapi, 'content', { lastScenario: scenario });
        } catch (error) {
          result.skipped += 1;
          result.errors.push(error instanceof Error ? error.message : String(error));
        }
      }

      await persistJobStatus(strapi, 'content', {
        lastSuccessAt: new Date().toISOString(),
        lastError: result.errors.length > 0 ? result.errors.join(' | ').slice(0, 1000) : null,
      });

      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      await persistJobStatus(strapi, 'content', { lastError: message });
      throw error;
    }
  });
}

export async function testContentAutomation(strapi: Core.Strapi): Promise<AiContentAutomationTestResult> {
  const settings = await getAiAutomationSettings(strapi);
  const modelPool = getEnabledModels(settings);
  if (modelPool.length === 0) {
    throw new Error('No enabled AI provider/model configured for content generation');
  }

  const availableCategories = await fetchContentCategories(strapi, settings.content.categoryDocumentIds);
  const category = availableCategories[0] ?? (await fetchAnyPublishedCategory(strapi)) ?? GENERIC_CONTENT_CATEGORY;
  const scenarios = parseScenarioPrompt(settings.content.scenarioPrompt);
  const scenario = pickScenario(scenarios, settings.content.lastScenario);

  const selectedModel = pickRandom(modelPool);
  const generated = await generateStructuredContent(selectedModel, settings.content.contentPrompt, category, scenario);
  const images: Array<{ provider: ImageSearchProvider; url: string; alt?: string }> = [];

  if (hasImageProviderCredentials(settings, settings.content.imageProvider)) {
    for (const query of generated.imageSearchQueries) {
      const batch = await searchRemoteImages(settings, query, 6, settings.content.imageProvider).catch(() => []);
      for (const item of batch) {
        if (!images.some((existing) => existing.url === item.url)) {
          images.push({
            provider: item.provider,
            url: item.url,
            alt: item.alt,
          });
        }
        if (images.length >= 6) {
          break;
        }
      }
      if (images.length >= 6) {
        break;
      }
    }
  }

  await persistJobStatus(strapi, 'content', { lastScenario: scenario });

  return {
    job: 'content',
    provider: selectedModel.provider,
    model: selectedModel.model,
    category: {
      documentId: category.documentId,
      name: category.name,
      slug: category.slug,
    },
    images,
    preview: {
      selectedScenario: generated.selectedScenario,
      title: generated.title,
      excerpt: generated.excerpt,
      bodyText: generated.bodyText,
      imageSearchQueries: generated.imageSearchQueries,
      relatedTags: generated.relatedTags,
      mediaMode: generated.mediaMode === 'gallery' ? 'gallery' : 'body',
    },
    warnings: hasImageProviderCredentials(settings, settings.content.imageProvider)
      ? []
      : ['Missing image search provider credentials. Real content run will create text-only posts.'],
  };
}

async function generateCommentForPost(
  selected: SelectedModel,
  post: PostEntry,
  replyTarget: CommentEntry | null,
  commentPrompt: string,
  replyPrompt: string
) {
  const instructions =
    `${replyTarget ? replyPrompt : commentPrompt}\n` +
    'Tra ve duy nhat noi dung comment, khong markdown. Neu khong that su can thiet thi khong dung emoji.';

  const prompt = replyTarget
    ? `Bai viet: "${post.title}"\nNoi dung bai: "${stripHtml(post.content).slice(0, 800)}"\nComment can reply: "${stripHtml(
        replyTarget.content
      )}" cua ${replyTarget.authorName}\nViet 1 reply ngan 1-2 cau.`
    : `Bai viet: "${post.title}"\nNoi dung bai: "${stripHtml(post.content).slice(
        0,
        800
      )}"\nViet 1 comment ngan 1-2 cau, dung voi cam xuc/chu de bai viet.`;

  const text = await generateProviderText(selected, {
    systemPrompt: instructions,
    userPrompt: prompt,
    maxTokens: 120,
  });
  return collapseWhitespace(text) || null;
}

async function generateCommentForPostWithFallback(
  modelPool: SelectedModel[],
  post: PostEntry,
  replyTarget: CommentEntry | null,
  commentPrompt: string,
  replyPrompt: string
) {
  const errors: string[] = [];
  const attempts = Math.min(3, Math.max(1, modelPool.length > 0 ? 3 : 1));
  const shuffled = shuffle(modelPool);

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const selectedModel = shuffled[attempt % shuffled.length];
    try {
      const generated = await generateCommentForPost(selectedModel, post, replyTarget, commentPrompt, replyPrompt);
      return { selectedModel, generated, errors };
    } catch (error) {
      errors.push(
        `attempt ${attempt + 1}/${attempts} ${selectedModel.provider}/${selectedModel.model}: ${
          error instanceof Error ? error.message : String(error)
        }`
      );
    }
  }

  throw new Error(
    errors[errors.length - 1] ?? `Unable to generate comment after ${attempts} attempts`
  );
}

export async function runCommentAutomation(strapi: Core.Strapi): Promise<AiAutomationRunResult> {
  return withJobLock(strapi, 'comments', async () => {
    const settings = await getAiAutomationSettings(strapi);
    const result: AiAutomationRunResult = {
      job: 'comments',
      createdComments: 0,
      skipped: 0,
      errors: [],
    };

    const nowIso = new Date().toISOString();
    await persistJobStatus(strapi, 'comments', { lastRunAt: nowIso, lastError: null });

    try {
      const users = await fetchSeededUsers(strapi);
      const modelPool = getEnabledModels(settings);
      if (users.length === 0) {
        throw new Error('No seeded users available for AI comments');
      }
      if (modelPool.length === 0) {
        throw new Error('No enabled AI provider/model configured for comment generation');
      }

      const posts = await fetchPublishedPosts(strapi, Math.max(settings.comments.commentsPerRun * 3, 20));

      if (posts.length === 0) {
        throw new Error('No published posts available for AI comments');
      }

      const shuffledPosts = [...posts].sort(() => Math.random() - 0.5).slice(0, settings.comments.commentsPerRun);
      for (const post of shuffledPosts) {
        try {
          const authorPool = users.filter((user) => user.id !== post.author?.id);
          const actor = pickRandom(authorPool.length > 0 ? authorPool : users);
          const existingComments = (await strapi.db.query('api::comment.comment').findMany({
            where: { targetType: 'post', targetDocumentId: post.documentId },
            limit: 50,
          })) as CommentEntry[];
          const canReply = settings.comments.allowReplies && existingComments.length > 0;
          const replyTarget = canReply && Math.random() < 0.5 ? pickRandom(existingComments) : null;
          const {
            generated,
            errors: generationErrors,
          } = await generateCommentForPostWithFallback(
            modelPool,
            post,
            replyTarget,
            settings.comments.commentPrompt,
            settings.comments.replyPrompt
          );
          result.errors.push(...generationErrors);
          if (!generated) {
            result.skipped += 1;
            continue;
          }

          const data: Record<string, unknown> = {
            authorName: actor.username,
            authorEmail: actor.email,
            content: generated,
            targetType: 'post',
            targetDocumentId: post.documentId,
          };
          if (replyTarget) {
            data.parent = replyTarget.id;
          }

          await strapi.db.query('api::comment.comment').create({ data });
          result.createdComments = (result.createdComments ?? 0) + 1;
        } catch (error) {
          result.skipped += 1;
          result.errors.push(error instanceof Error ? error.message : String(error));
        }
      }

      await persistJobStatus(strapi, 'comments', {
        lastSuccessAt: new Date().toISOString(),
        lastError: result.errors.length > 0 ? result.errors.join(' | ').slice(0, 1000) : null,
      });

      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      await persistJobStatus(strapi, 'comments', { lastError: message });
      throw error;
    }
  });
}

export async function testCommentAutomation(strapi: Core.Strapi): Promise<AiCommentAutomationTestResult> {
  const settings = await getAiAutomationSettings(strapi);
  const modelPool = getEnabledModels(settings);
  if (modelPool.length === 0) {
    throw new Error('No enabled AI provider/model configured for comment generation');
  }

  const posts = await fetchPublishedPosts(strapi, 20);
  if (posts.length === 0) {
    throw new Error('No published posts available for AI comments');
  }

  const post = pickRandom(posts);
  const existingComments = (await strapi.db.query('api::comment.comment').findMany({
    where: { targetType: 'post', targetDocumentId: post.documentId },
    limit: 50,
  })) as CommentEntry[];
  const canReply = settings.comments.allowReplies && existingComments.length > 0;
  const replyTarget = canReply && Math.random() < 0.5 ? pickRandom(existingComments) : null;
  const selectedModel = pickRandom(modelPool);
  const preview = await generateCommentForPost(
    selectedModel,
    post,
    replyTarget,
    settings.comments.commentPrompt,
    settings.comments.replyPrompt
  );
  if (!preview) {
    throw new Error('AI returned empty comment preview');
  }

  return {
    job: 'comments',
    provider: selectedModel.provider,
    model: selectedModel.model,
    post: {
      documentId: post.documentId,
      title: post.title,
    },
    replyMode: replyTarget ? 'reply' : 'top-level',
    preview,
  };
}

export async function runDueAiAutomation(strapi: Core.Strapi) {
  const settings = await getAiAutomationSettings(strapi);
  const now = new Date();
  const timeZone = settings.timezone || DEFAULT_TIMEZONE;
  const minuteKey = getMinuteKey(now, timeZone);
  const dueJobs: Array<'content' | 'comments'> = [];

  if (
    settings.content.enabled &&
    cronMatches(settings.content.cron, now, timeZone) &&
    (!settings.content.lastRunAt || getMinuteKey(new Date(settings.content.lastRunAt), timeZone) !== minuteKey)
  ) {
    dueJobs.push('content');
  }

  if (
    settings.comments.enabled &&
    cronMatches(settings.comments.cron, now, timeZone) &&
    (!settings.comments.lastRunAt || getMinuteKey(new Date(settings.comments.lastRunAt), timeZone) !== minuteKey)
  ) {
    dueJobs.push('comments');
  }

  for (const job of dueJobs) {
    try {
      if (job === 'content') {
        await runContentAutomation(strapi);
      } else {
        await runCommentAutomation(strapi);
      }
    } catch (error) {
      console.error(`[ai-automation] ${job} failed:`, error);
    }
  }
}
