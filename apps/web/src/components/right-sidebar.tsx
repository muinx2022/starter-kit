import Link from "next/link";
import { getTopPosts, getTopTags, type Category } from "@/lib/strapi";

const COLOR_CLASSES = [
  "bg-rose-400",
  "bg-orange-400",
  "bg-amber-400",
  "bg-lime-500",
  "bg-emerald-500",
  "bg-cyan-500",
  "bg-sky-400",
  "bg-blue-400",
  "bg-indigo-400",
  "bg-violet-400",
  "bg-fuchsia-400",
  "bg-pink-400",
];

function getColorBySlug(slug: string) {
  const hash = Array.from(slug).reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  return COLOR_CLASSES[hash % COLOR_CLASSES.length];
}

type RightSidebarProps = {
  categories?: Category[];
};

export async function RightSidebar({ categories = [] }: RightSidebarProps) {
  let posts: Awaited<ReturnType<typeof getTopPosts>> = [];
  let tags: Awaited<ReturnType<typeof getTopTags>> = [];

  try {
    const [topPosts, topTags] = await Promise.all([
      getTopPosts(5),
      getTopTags(16),
    ]);
    posts = topPosts;
    tags = topTags;
  } catch {
    // Strapi unreachable - render with empty data
  }

  return (
    <div className="space-y-4">
      {categories.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
          <h3 className="border-b border-gray-100 px-4 py-3 text-sm font-semibold text-gray-800 dark:border-gray-700 dark:text-gray-100">Danh mục</h3>
          <nav className="flex flex-col gap-1 p-2">
            <Link
              href="/"
              className="flex items-center gap-2.5 rounded-md px-3 py-2.5 text-sm text-gray-600 transition-colors hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-gray-100"
            >
              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-gray-400 text-white">
                <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                  <polyline points="9 22 9 12 15 12 15 22" />
                </svg>
              </span>
              <span className="font-medium">Trang chủ</span>
            </Link>
            {categories.map((category) => (
              <Link
                key={category.documentId}
                href={`/c/${category.slug}`}
                className="flex items-center gap-2.5 rounded-md px-3 py-2.5 text-sm text-gray-600 transition-colors hover:bg-gray-50 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-gray-100"
              >
                <span
                  className={`flex h-5 w-5 shrink-0 items-center justify-center rounded text-[10px] font-bold text-white ${getColorBySlug(category.slug)}`}
                >
                  {category.name[0]?.toUpperCase()}
                </span>
                <span className="font-medium">{category.name}</span>
              </Link>
            ))}
          </nav>
        </div>
      )}

      <div className="rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
        <h3 className="flex items-center gap-1.5 border-b border-gray-100 px-4 py-3 text-sm font-semibold text-gray-800 dark:border-gray-700 dark:text-gray-100">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
          </svg>
          Bài viết nổi bật
        </h3>
        <div className="flex flex-col gap-1 p-2">
          {posts.length > 0 ? (
            posts.map((post, index) => (
              <Link
                key={post.documentId}
                href={`/p/${post.slug}--${post.documentId}`}
                className="group flex items-start gap-3 rounded-md px-2 py-2.5 transition-colors hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded bg-gray-100 text-xs font-semibold text-gray-400 dark:bg-gray-700 dark:text-gray-500">
                  {index + 1}
                </span>
                <div className="min-w-0 flex-1">
                  <h4 className="line-clamp-2 text-sm leading-snug text-gray-700 transition-colors group-hover:text-gray-900 dark:text-gray-300 dark:group-hover:text-gray-100">
                    {post.title}
                  </h4>
                  <div className="mt-1 flex items-center gap-2 text-xs text-gray-400">
                    <span>{post.likesCount ?? 0} thích</span>
                    <span>·</span>
                    <span>{post.commentsCount ?? 0} bình luận</span>
                  </div>
                </div>
              </Link>
            ))
          ) : (
            <div className="py-4 text-center text-sm text-gray-400">Chưa có bài viết</div>
          )}
        </div>
      </div>

      {tags.length > 0 && (
        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-800">
          <h3 className="mb-3 text-sm font-semibold text-gray-800 dark:text-gray-100">Xu hướng</h3>
          <div className="flex flex-wrap gap-1.5">
            {tags.map((tag) => (
              <Link
                key={tag.documentId}
                href={`/t/${tag.slug}`}
                className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-1 text-xs text-gray-600 transition-colors hover:bg-gray-200 hover:text-gray-800 dark:bg-gray-700 dark:text-gray-300 dark:hover:bg-gray-600 dark:hover:text-gray-100"
              >
                #{tag.name}
              </Link>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
