"use client";

import { useCallback, useEffect, useState } from "react";
import { Trash2, CheckCheck, Clock, XCircle, ChevronDown, ChevronRight, ExternalLink } from "lucide-react";
import { toast } from "@/components/ui/app-toast";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { IconAction } from "@/components/icon-action";
import { PaginationControls } from "@/components/pagination-controls";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  deleteReport,
  listReports,
  updateReportStatus,
  type PaginationMeta,
  type ReportItem,
  type ReportStatus,
} from "@/lib/admin-api";

function formatDate(value?: string | null) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

const STATUS_STYLES: Record<ReportStatus, string> = {
  pending: "bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400",
  reviewed: "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400",
  dismissed: "bg-zinc-100 text-zinc-500 dark:bg-zinc-800 dark:text-zinc-400",
};

const STATUS_LABELS: Record<ReportStatus, string> = {
  pending: "Chờ xử lý",
  reviewed: "Đã xem xét",
  dismissed: "Bỏ qua",
};

type ReportGroup = {
  targetType: string;
  targetDocumentId: string;
  targetTitle?: string | null;
  targetSlug?: string | null;
  reports: ReportItem[];
  pendingCount: number;
};

function groupReports(reports: ReportItem[]): ReportGroup[] {
  const map = new Map<string, ReportGroup>();
  for (const r of reports) {
    const key = `${r.targetType}:${r.targetDocumentId}`;
    if (!map.has(key)) {
      map.set(key, {
        targetType: r.targetType,
        targetDocumentId: r.targetDocumentId,
        targetTitle: r.targetTitle,
        targetSlug: r.targetSlug,
        reports: [],
        pendingCount: 0,
      });
    }
    const group = map.get(key)!;
    group.reports.push(r);
    if (r.status === "pending") group.pendingCount++;
    if (r.targetTitle) group.targetTitle = r.targetTitle;
    if (r.targetSlug) group.targetSlug = r.targetSlug;
  }
  return Array.from(map.values()).sort((a, b) => b.pendingCount - a.pendingCount || b.reports.length - a.reports.length);
}

const WEB_URL = process.env.NEXT_PUBLIC_WEB_URL ?? "";

export function ReportsManager() {
  const [rows, setRows] = useState<ReportItem[]>([]);
  const [pagination, setPagination] = useState<PaginationMeta>({ page: 1, pageSize: 10, pageCount: 1, total: 0 });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [statusFilter, setStatusFilter] = useState<"all" | ReportStatus>("all");
  const [targetTypeFilter, setTargetTypeFilter] = useState("all");
  const [processingId, setProcessingId] = useState<number | null>(null);
  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(new Set());

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await listReports(page, 10, { status: statusFilter, targetType: targetTypeFilter });
      setRows(result.data);
      setPagination(result.pagination);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load reports");
    } finally {
      setLoading(false);
    }
  }, [page, statusFilter, targetTypeFilter]);

  useEffect(() => { void load(); }, [load]);

  const toggleExpand = (key: string) => {
    setExpandedKeys((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  const onUpdateStatus = async (item: ReportItem, status: ReportStatus) => {
    if (item.status === status) return;
    setProcessingId(item.id);
    try {
      const updated = await updateReportStatus(item.id, status);
      setRows((prev) => prev.map((r) => r.id === item.id ? { ...r, status: updated.status } : r));
      toast({ title: `Cập nhật thành "${STATUS_LABELS[status]}"`, variant: "success" });
    } catch (err) {
      toast({ title: "Cập nhật thất bại", description: err instanceof Error ? err.message : undefined, variant: "error" });
    } finally {
      setProcessingId(null);
    }
  };

  const onDelete = async (item: ReportItem) => {
    if (!confirm(`Xoá báo cáo #${item.id}?`)) return;
    try {
      await deleteReport(item.id);
      toast({ title: "Đã xoá báo cáo", variant: "success" });
      setRows((prev) => prev.filter((r) => r.id !== item.id));
    } catch (err) {
      toast({ title: "Xoá thất bại", description: err instanceof Error ? err.message : undefined, variant: "error" });
    }
  };

  const groups = groupReports(rows);

  return (
    <Card>
      <CardHeader>
        <div>
          <CardTitle className="text-xl font-semibold tracking-tight">Reports</CardTitle>
          <p className="text-sm text-muted-foreground">Quản lý các báo cáo từ người dùng · {groups.length} mục bị báo cáo</p>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-3 flex flex-wrap gap-2">
          <select
            className="h-10 rounded-md border bg-background px-3 text-sm"
            value={statusFilter}
            onChange={(e) => { setPage(1); setStatusFilter(e.target.value as "all" | ReportStatus); }}
          >
            <option value="all">Tất cả trạng thái</option>
            <option value="pending">Chờ xử lý</option>
            <option value="reviewed">Đã xem xét</option>
            <option value="dismissed">Bỏ qua</option>
          </select>
          <select
            className="h-10 rounded-md border bg-background px-3 text-sm"
            value={targetTypeFilter}
            onChange={(e) => { setPage(1); setTargetTypeFilter(e.target.value); }}
          >
            <option value="all">Tất cả loại</option>
            <option value="post">Post</option>
            <option value="comment">Comment</option>
            <option value="user">User</option>
          </select>
        </div>

        {loading && <p className="text-sm text-muted-foreground py-2">Đang tải...</p>}
        {error && <p className="text-sm text-destructive py-2">{error}</p>}

        <div className="rounded-md border divide-y">
          {groups.length === 0 && !loading && (
            <div className="py-10 text-center text-sm text-muted-foreground">Không có báo cáo nào.</div>
          )}

          {groups.map((group) => {
            const key = `${group.targetType}:${group.targetDocumentId}`;
            const expanded = expandedKeys.has(key);
            const postUrl = group.targetType === "post" && group.targetSlug
              ? `${WEB_URL}/p/${group.targetSlug}--${group.targetDocumentId}`
              : null;

            return (
              <div key={key}>
                {/* Group header row */}
                <div
                  className="flex items-center gap-3 px-4 py-3 hover:bg-muted/40 cursor-pointer select-none"
                  onClick={() => toggleExpand(key)}
                >
                  <span className="text-muted-foreground">
                    {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                  </span>

                  <span className="capitalize text-xs font-medium text-muted-foreground bg-muted px-2 py-0.5 rounded shrink-0">{group.targetType}</span>

                  <div className="min-w-0 flex-1">
                    {group.targetTitle ? (
                      <p className="text-sm font-medium truncate">{group.targetTitle}</p>
                    ) : (
                      <p className="font-mono text-xs text-muted-foreground truncate">{group.targetDocumentId}</p>
                    )}
                  </div>

                  {postUrl && (
                    <a
                      href={postUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      onClick={(e) => e.stopPropagation()}
                      className="text-muted-foreground hover:text-foreground transition-colors shrink-0"
                      title="Xem bài viết"
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                    </a>
                  )}

                  <div className="ml-auto flex items-center gap-2 shrink-0">
                    {group.pendingCount > 0 && (
                      <span className="inline-flex items-center rounded-full bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 px-2 py-0.5 text-xs font-semibold">
                        {group.pendingCount} chờ xử lý
                      </span>
                    )}
                    <span className="inline-flex items-center rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 px-2 py-0.5 text-xs font-medium">
                      {group.reports.length} báo cáo
                    </span>
                  </div>
                </div>

                {/* Expanded individual reports */}
                {expanded && (
                  <Table className="[&_td]:align-top">
                    <TableHeader>
                      <TableRow className="bg-muted/20">
                        <TableHead className="w-[50px] pl-12">ID</TableHead>
                        <TableHead>Lý do</TableHead>
                        <TableHead className="w-[180px]">Người báo cáo</TableHead>
                        <TableHead className="w-[140px]">Ngày tạo</TableHead>
                        <TableHead className="w-[120px]">Trạng thái</TableHead>
                        <TableHead className="w-[1%] text-right pr-4">Xử lý</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {group.reports.map((item) => (
                        <TableRow key={item.id} className="group bg-muted/10">
                          <TableCell className="pl-12 text-muted-foreground text-sm">{item.id}</TableCell>
                          <TableCell className="text-sm max-w-[260px]">{item.reason || <span className="text-muted-foreground italic">Không có lý do</span>}</TableCell>
                          <TableCell>
                            {item.reporter ? (
                              <div>
                                <p className="text-sm font-medium">{item.reporter.username}</p>
                                <p className="text-xs text-muted-foreground">{item.reporter.email}</p>
                              </div>
                            ) : (
                              <span className="text-muted-foreground text-sm">Ẩn danh</span>
                            )}
                          </TableCell>
                          <TableCell className="text-sm">{formatDate(item.createdAt)}</TableCell>
                          <TableCell>
                            <span className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_STYLES[item.status]}`}>
                              {STATUS_LABELS[item.status]}
                            </span>
                          </TableCell>
                          <TableCell className="text-right pr-4">
                            <div className="ml-auto flex w-fit gap-1 opacity-100 pointer-events-auto transition-opacity duration-150 md:opacity-0 md:pointer-events-none md:group-hover:opacity-100 md:group-hover:pointer-events-auto">
                              <IconAction label="Đã xem xét" icon={<CheckCheck />} onClick={() => onUpdateStatus(item, "reviewed")} variant={item.status === "reviewed" ? "default" : "outline"} size="icon-xs" disabled={processingId === item.id} />
                              <IconAction label="Chờ xử lý" icon={<Clock />} onClick={() => onUpdateStatus(item, "pending")} variant={item.status === "pending" ? "default" : "outline"} size="icon-xs" disabled={processingId === item.id} />
                              <IconAction label="Bỏ qua" icon={<XCircle />} onClick={() => onUpdateStatus(item, "dismissed")} variant={item.status === "dismissed" ? "secondary" : "outline"} size="icon-xs" disabled={processingId === item.id} />
                              <IconAction label="Xoá" icon={<Trash2 />} onClick={() => onDelete(item)} variant="destructive" size="icon-xs" />
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </div>
            );
          })}
        </div>

        <PaginationControls
          page={pagination.page}
          pageCount={pagination.pageCount}
          total={pagination.total}
          onPageChange={setPage}
        />
      </CardContent>
    </Card>
  );
}
