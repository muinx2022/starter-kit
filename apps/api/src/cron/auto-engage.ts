import OpenAI from 'openai';
import type { Core } from '@strapi/strapi';

const BATCH_SIZE = parseInt(process.env.CRON_POSTS_PER_RUN ?? '5', 10);
const MAX_IMAGES = 5;
const MODEL = process.env.OPENAI_AUTO_ENGAGE_MODEL ?? 'gpt-4.1-mini';

interface TiptapNode {
  type?: string;
  text?: string;
  attrs?: Record<string, unknown>;
  content?: TiptapNode[];
}

type OpenAIInputContent =
  | { type: 'input_text'; text: string }
  | { type: 'input_image'; image_url: string; detail: 'auto' };

function collapseWhitespace(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

function stripHtml(value: string): string {
  return collapseWhitespace(value.replace(/<[^>]+>/g, ' '));
}

function buildContentBlocks(
  node: TiptapNode,
  imageCount: { value: number }
): OpenAIInputContent[] {
  const blocks: OpenAIInputContent[] = [];

  if (!node) return blocks;

  if (node.type === 'text' && node.text) {
    blocks.push({ type: 'input_text', text: node.text });
    return blocks;
  }

  if (node.type === 'image' && imageCount.value < MAX_IMAGES) {
    const src = node.attrs?.src as string | undefined;
    if (src?.startsWith('https://')) {
      blocks.push({ type: 'input_image', image_url: src, detail: 'auto' });
      imageCount.value++;
    }
    return blocks;
  }

  if (node.type === 'video') {
    blocks.push({ type: 'input_text', text: '[video]' });
    return blocks;
  }

  if (node.content) {
    for (const child of node.content) {
      blocks.push(...buildContentBlocks(child, imageCount));
    }
  }

  return blocks;
}

function extractPlainText(node: TiptapNode | null | undefined): string {
  if (!node) return '';

  if (node.type === 'text' && node.text) {
    return node.text;
  }

  if (node.type === 'video') {
    return ' [video] ';
  }

  const parts = (node.content ?? [])
    .map((child) => extractPlainText(child))
    .filter(Boolean);

  if (!parts.length) return '';

  const joined = parts.join(node.type === 'paragraph' || node.type === 'heading' ? '\n' : ' ');
  return collapseWhitespace(joined);
}

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

export async function autoEngage(strapi: Core.Strapi) {
  const log = (msg: string) => console.log(`[cron:autoEngage] ${msg}`);
  const apiKey = process.env.OPENAI_API_KEY;

  if (!apiKey) {
    log('OPENAI_API_KEY not set - skipping.');
    return;
  }

  const seededUsers = (await strapi.db
    .query('plugin::users-permissions.user')
    .findMany({
      where: { isSeeded: true, blocked: false, confirmed: true },
      limit: 200,
    })) as Array<{ id: number; documentId: string; username: string; email: string }>;

  if (!seededUsers.length) {
    log('No seeded users (isSeeded=true) found - skipping.');
    return;
  }

  const store = strapi.store({ type: 'core', name: 'cron', key: 'autoEngageOffset' });
  const currentOffset = ((await store.get({})) as number) ?? 0;

  const posts = (await strapi.db.query('api::post.post').findMany({
    where: { publishedAt: { $notNull: true } },
    populate: ['author', 'images'],
    orderBy: { createdAt: 'asc' },
    limit: BATCH_SIZE,
    offset: currentOffset,
  })) as Array<{
    id: number;
    documentId: string;
    title: string;
    content: TiptapNode;
    images: Array<{ url: string }>;
    author?: { id: number; documentId: string; username: string };
  }>;

  const nextOffset = posts.length < BATCH_SIZE ? 0 : currentOffset + BATCH_SIZE;
  await store.set({ value: nextOffset });
  log(`Processing ${posts.length} posts (offset ${currentOffset} -> ${nextOffset})`);

  if (!posts.length) {
    log('No published posts found.');
    return;
  }

  const openai = new OpenAI({ apiKey });

  for (const post of posts) {
    try {
      await engagePost(strapi, openai, post, seededUsers, log);
    } catch (err) {
      log(`Error on post ${post.documentId}: ${err}`);
    }
  }
}

async function engagePost(
  strapi: Core.Strapi,
  openai: OpenAI,
  post: {
    id: number;
    documentId: string;
    title: string;
    content: TiptapNode;
    images: Array<{ url: string }>;
    author?: { id: number; documentId: string; username: string };
  },
  seededUsers: Array<{ id: number; documentId: string; username: string; email: string }>,
  log: (msg: string) => void
) {
  const candidates = seededUsers.filter((u) => u.id !== post.author?.id);
  if (!candidates.length) return;
  const actor = pickRandom(candidates);

  const existingComments = (await strapi.db.query('api::comment.comment').findMany({
    where: { targetType: 'post', targetDocumentId: post.documentId },
    limit: 50,
  })) as Array<{ id: number; documentId: string; authorName: string; content: string }>;

  const shouldReply = existingComments.length > 0 && Math.random() < 0.5;
  const parentComment = shouldReply ? pickRandom(existingComments) : null;

  const generatedComment = await generateComment(openai, post, parentComment);

  if (!generatedComment) {
    log(`Skipped comment for post ${post.documentId} - empty AI response`);
  } else {
    const commentData: Record<string, unknown> = {
      authorName: actor.username,
      authorEmail: actor.email,
      content: generatedComment,
      targetType: 'post',
      targetDocumentId: post.documentId,
    };

    if (parentComment) {
      commentData.parent = parentComment.id;
    }

    await strapi.db.query('api::comment.comment').create({ data: commentData });
    log(`Commented on post ${post.documentId} as "${actor.username}"${parentComment ? ' (reply)' : ''}`);
  }

  const alreadyLiked = await strapi.db.query('api::interaction.interaction').findOne({
    where: {
      actionType: 'like',
      targetType: 'post',
      targetDocumentId: post.documentId,
      user: actor.id,
    },
  });

  if (!alreadyLiked) {
    await strapi.db.query('api::interaction.interaction').create({
      data: {
        actionType: 'like',
        targetType: 'post',
        targetDocumentId: post.documentId,
        user: actor.id,
      },
    });
    log(`Liked post ${post.documentId} as "${actor.username}"`);
  }

  if (post.author && post.author.id !== actor.id) {
    const alreadyFollowed = await strapi.db.query('api::interaction.interaction').findOne({
      where: {
        actionType: 'follow',
        targetType: 'user',
        targetDocumentId: post.author.documentId,
        user: actor.id,
      },
    });

    if (!alreadyFollowed) {
      await strapi.db.query('api::interaction.interaction').create({
        data: {
          actionType: 'follow',
          targetType: 'user',
          targetDocumentId: post.author.documentId,
          user: actor.id,
        },
      });
      log(`Followed user "${post.author.username}" as "${actor.username}"`);
    }
  }
}

async function generateComment(
  openai: OpenAI,
  post: {
    title: string;
    content: TiptapNode;
    images: Array<{ url: string }>;
  },
  parentComment: { authorName: string; content: string } | null
): Promise<string | null> {
  const imageCount = { value: 0 };
  const richTextBlocks = buildContentBlocks(post.content ?? {}, imageCount);
  const plainPostText = extractPlainText(post.content);
  const hasVideoInContent = richTextBlocks.some(
    (b) => b.type === 'input_text' && b.text === '[video]'
  );

  for (const img of post.images ?? []) {
    if (imageCount.value >= MAX_IMAGES) break;
    if (img.url?.startsWith('https://')) {
      richTextBlocks.push({ type: 'input_image', image_url: img.url, detail: 'auto' });
      imageCount.value++;
    }
  }

  const hasImages = imageCount.value > 0;
  const hasVideo = hasVideoInContent;
  const hasBodyText = plainPostText.length > 0;
  const titleOnly = !hasBodyText && !hasImages && !hasVideo;
  const sparseInput = !hasBodyText && (hasImages || hasVideo);
  const sanitizedParentComment = parentComment
    ? {
        authorName: collapseWhitespace(parentComment.authorName),
        content: stripHtml(parentComment.content),
      }
    : null;

  const systemPrompt =
    'Ban dang viet comment gia lap cho mang xa hoi bang tieng Viet. ' +
    'Hay viet nhu nguoi dung that: ngan, tu nhien, khong formal, khong sao rong, khong emoji. ' +
    'Chi duoc dung thong tin that su co trong input. Khong bia them boi canh, dia diem, thoi gian, nhan vat, trai nghiem hay chi tiet khong thay ro. ' +
    'Neu input it thong tin thi phai viet than trong va ngan hon. ' +
    'Duoc phep viet nhu mot nguoi xem that tung di qua, tung ghe, tung biet ve dia diem/chu de do, nhung chi noi rat ngan gon va doi thuong. ' +
    'Neu nhin thay ro anh thi duoc khen nhung thu truc quan nhu bo cuc, mau sac, anh sang, goc chup, khong khi, do net, cam giac de xem. ' +
    'Khong duoc tu mot anh ma suy dien ra ca mot hanh trinh, mot qua trinh, mot cau chuyen lon, hay y nghia qua muc.';

  const inputConstraint = titleOnly
    ? 'Input gan nhu chi co tieu de. Hay viet mot nhan xet rat ngan, an toan, khong bia noi dung cu the cua bai.'
    : !hasImages && !hasVideo
      ? 'Bai viet nay khong co anh hoac video. Khong duoc nhac den anh/video.'
      : sparseInput
        ? 'Input hien chi co tieu de va/hoac hinh anh, gan nhu khong co body text. Chi duoc phan ung theo nhung gi thay ro tu tieu de/hinh. Co the noi kieu "da tung ghe", "co biet cho nay", hoac khen bo cuc/anh dep neu nhin thay ro, nhung khong duoc suy dien thanh ca qua trinh hay cau chuyen phia sau.'
        : 'Neu anh khong du ro de ket luan, hay giu nhan xet o muc chung chung thay vi doan.';

  let userContent: OpenAIInputContent[];

  if (sanitizedParentComment) {
    userContent = [
      {
        type: 'input_text',
        text:
          `Tieu de bai viet: "${post.title}"\n` +
          `Binh luan can tra loi: "${sanitizedParentComment.content}" - ${sanitizedParentComment.authorName}\n\n` +
          `Rang buoc: ${inputConstraint}\n` +
          'Viet reply dai 1-3 cau, kieu nguoi dung that dang trao doi tren mang. ' +
          'Neu input mo ho thi giu giong vua phai. Co the noi kieu "minh cung tung ghe", "nhin bo cuc dep", "anh len mau on" neu hop input. ' +
          'Khong tung ho, khong nang tam thanh "rat sau sac", "dam tinh lich su", "qua y nghia", va khong bien mot hinh anh thanh ca mot qua trinh neu khong co can cu. ' +
          'Khong dong vai tac gia bai viet. Khong lap lai nguyen y nguoi truoc. Chi tra ve noi dung comment.',
      },
    ];
  } else {
    const intro: OpenAIInputContent = {
      type: 'input_text',
      text:
        `Tieu de bai viet: "${post.title}"\n` +
        `Text trong bai: ${hasBodyText ? `"${plainPostText.slice(0, 1200)}"` : '[khong co hoac rat it]'}\n` +
        'Duoi day la noi dung va anh dinh kem cua bai:',
    };

    const outro: OpenAIInputContent = {
      type: 'input_text',
      text:
        `\n\nRang buoc: ${inputConstraint}\n` +
        'Viet 1 comment voi tu cach nguoi doc, khong phai tac gia. ' +
        'Comment phai dai 1-3 cau. Neu du lieu it thi van giu 1-3 cau nhung phan ung vua phai, khong phong dai. ' +
        'Co the viet theo kieu da tung ghe qua, tung biet den noi nay, hoac khen nhung diem nhin thay ro tu anh nhu bo cuc, anh sang, mau, goc chup. ' +
        'Khong duoc tu nhay thanh khen lon, suy ton, gan chat "lich su", "sau sac", "day thong diep", hoac noi ve ca mot hanh trinh, mot qua trinh neu input khong du can cu. ' +
        'Khong duoc viet kieu ket bai cua tac gia nhu "cam on moi nguoi", "minh se co gang". ' +
        'Khong dung emoji. Chi tra ve noi dung comment.',
    };

    userContent =
      richTextBlocks.length > 0
        ? [intro, ...richTextBlocks, outro]
        : [
            {
              type: 'input_text',
              text:
                `Tieu de bai viet: "${post.title}". ` +
                `Text trong bai: ${hasBodyText ? `"${plainPostText.slice(0, 1200)}". ` : ''}` +
                `Rang buoc: ${inputConstraint} ` +
                'Viet 1 comment dai 1-3 cau voi tu cach nguoi doc, khong phai tac gia. ' +
                'Neu input mo ho thi chi nen nhan xet muc do vua phai. Co the nhac kieu da tung ghe, tung biet, hoac khen anh dep/bo cuc dep neu input cho phep. ' +
                'Khong tung ho, khong gan y nghia lon, va khong bien mot hinh anh thanh ca mot qua trinh neu khong co can cu. ' +
                'Khong bia them chi tiet ngoai input. Khong dung emoji. Chi tra ve noi dung comment.',
            },
          ];
  }

  const response = await openai.responses.create({
    model: MODEL,
    instructions: systemPrompt,
    input: [
      {
        role: 'user',
        content: userContent,
      },
    ],
    max_output_tokens: 120,
  });

  return response.output_text.trim() || null;
}
