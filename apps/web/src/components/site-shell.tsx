import Link from "next/link";
import { ReactNode } from "react";
import { RightSidebar } from "@/components/right-sidebar";
import { SiteHeader } from "@/components/site-header";
import { LoginModal } from "@/components/login-modal";
import { LeftSidebar } from "@/components/left-sidebar";
import { MobileRightDrawer } from "@/components/mobile-right-drawer";
import { getFooterPages, getTopLevelCategories } from "@/lib/strapi";

export async function SiteShell({ children }: { children: ReactNode }) {
  let categories: Awaited<ReturnType<typeof getTopLevelCategories>> = [];
  let footerPages: Awaited<ReturnType<typeof getFooterPages>> = [];

  try {
    [categories, footerPages] = await Promise.all([
      getTopLevelCategories(),
      getFooterPages(),
    ]);
  } catch {
    categories = [];
    footerPages = [];
  }

  return (
    <>
      <SiteHeader />
      <LoginModal />

      <div id="mobile-left-drawer" className="mobile-left-drawer md:hidden" aria-modal="true" role="dialog">
        <a href="#" className="mobile-left-overlay" aria-label="Dong menu" />
        <aside className="mobile-left-panel">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-base font-semibold text-gray-800 dark:text-gray-100">Danh muc</h3>
            <a href="#" className="rounded-md p-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700" aria-label="Dong">
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 6 6 18" />
                <path d="m6 6 12 12" />
              </svg>
            </a>
          </div>
          <LeftSidebar categories={categories} />
        </aside>
      </div>

      <MobileRightDrawer>
        <RightSidebar categories={categories} />
      </MobileRightDrawer>

      <div className="container mx-auto flex-1 px-4 py-6">
        <div className="grid grid-cols-1 gap-6 md:grid-cols-12">
          <main className="space-y-4 md:col-span-8">{children}</main>

          <aside className="hidden space-y-4 md:col-span-4 md:block">
            <RightSidebar categories={categories} />
          </aside>
        </div>
      </div>

      {footerPages.length > 0 && (
        <div className="fixed bottom-4 right-4 z-40 hidden md:flex items-center gap-2 rounded-full border border-gray-200 bg-white/90 backdrop-blur-sm px-3 py-1.5 shadow-sm text-xs text-gray-500 dark:border-gray-700 dark:bg-gray-800/90 dark:text-gray-400">
          {footerPages.map((page, idx) => (
            <span key={page.documentId} className="flex items-center gap-2">
              {idx > 0 && <span className="text-gray-300 dark:text-gray-600">|</span>}
              <Link
                href={`/page/${page.slug}`}
                className="font-medium text-gray-600 transition-colors hover:text-gray-900 hover:underline dark:text-gray-400 dark:hover:text-gray-200"
              >
                {page.title}
              </Link>
            </span>
          ))}
          <span className="text-gray-300 dark:text-gray-600">|</span>
          <span>&copy; {new Date().getFullYear()} Trekky</span>
        </div>
      )}
    </>
  );
}
