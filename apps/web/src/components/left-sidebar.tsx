import Link from "next/link";
import { Home } from "lucide-react";
import { getTopLevelCategories, type Category } from "@/lib/strapi";

const COLOR_CLASSES = [
  "bg-rose-500",
  "bg-orange-500",
  "bg-amber-500",
  "bg-lime-500",
  "bg-emerald-500",
  "bg-cyan-500",
  "bg-sky-500",
  "bg-blue-500",
  "bg-indigo-500",
  "bg-violet-500",
  "bg-fuchsia-500",
  "bg-pink-500",
];

function getColorBySlug(slug: string) {
  const hash = Array.from(slug).reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  return COLOR_CLASSES[hash % COLOR_CLASSES.length];
}

function getCategoryInitial(name: string) {
  const trimmed = name.trim();
  if (!trimmed) return "C";
  return trimmed[0]!.toUpperCase();
}

type LeftSidebarProps = {
  categories?: Category[];
};

export async function LeftSidebar({ categories: categoriesProp }: LeftSidebarProps = {}) {
  const categories = categoriesProp ?? (await getTopLevelCategories());

  return (
    <nav className="flex flex-col gap-1">
      <Link
        href="/"
        className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-gray-700 transition-colors hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-gray-100"
      >
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-gray-500 text-white">
          <Home className="h-3.5 w-3.5" />
        </span>
        <span className="flex-1 font-medium">Trang chủ</span>
      </Link>

      {categories && categories.length > 0 ? (
        categories.map((category) => (
          <Link
            key={category.documentId}
            href={`/c/${category.slug}`}
            className="group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-gray-700 transition-colors hover:bg-gray-100 hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-700 dark:hover:text-gray-100"
          >
            <span
              className={`flex h-6 w-6 items-center justify-center rounded-md text-xs font-semibold text-white ${getColorBySlug(category.slug)}`}
            >
              {getCategoryInitial(category.name)}
            </span>
            <span className="flex-1 font-medium">{category.name}</span>
          </Link>
        ))
      ) : (
        <div className="py-4 text-center text-sm text-gray-400">Chưa có danh mục</div>
      )}
    </nav>
  );
}
