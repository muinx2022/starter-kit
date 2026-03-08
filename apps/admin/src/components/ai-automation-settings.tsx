"use client";

import { useEffect, useMemo, useState } from "react";
import { Bot, Image as ImageIcon, MessageSquareText, RefreshCw, Save, Sparkles, X } from "lucide-react";
import { MultiSelectBox } from "@/components/multi-select-box";
import { toast } from "@/components/ui/app-toast";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  checkAiProviderConnection,
  getAiAutomationSettings,
  listAllCategories,
  runAiCommentCron,
  runAiContentCron,
  testAiComment,
  testAiContent,
  updateAiAutomationSettings,
  type AiAutomationRunResult,
  type AiAutomationSettings,
  type AiCommentAutomationTestResult,
  type AiContentAutomationTestResult,
  type CategoryItem,
} from "@/lib/admin-api";

const PROVIDER_MODELS = {
  openai: [
    { value: "gpt-4.1-mini", label: "gpt-4.1-mini" },
    { value: "gpt-4.1", label: "gpt-4.1" },
    { value: "gpt-4o-mini", label: "gpt-4o-mini" },
    { value: "gpt-4o", label: "gpt-4o" },
  ],
  anthropic: [
    { value: "claude-haiku-4-5", label: "claude-haiku-4-5" },
    { value: "claude-sonnet-4-6", label: "claude-sonnet-4-6" },
    { value: "claude-opus-4-6", label: "claude-opus-4-6" },
  ],
} as const;

const TABS = [
  { id: "ai", label: "AI", icon: Bot },
  { id: "images", label: "Image Search", icon: ImageIcon },
  { id: "content", label: "Content", icon: Sparkles },
  { id: "comments", label: "Comments", icon: MessageSquareText },
] as const;

type SettingsTab = (typeof TABS)[number]["id"];

function formatDate(value?: string | null) {
  if (!value) return "Never";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Never";
  return date.toLocaleString();
}

function RunResult({ result }: { result: AiAutomationRunResult | null }) {
  if (!result) return null;

  if (result.job === "content") {
    return (
      <p className="text-xs text-muted-foreground">
        Posts: {result.createdPosts ?? 0}, uploaded images: {result.uploadedImages ?? 0}, body images:{" "}
        {result.embeddedBodyImages ?? 0}, gallery images: {result.galleryImages ?? 0}, skipped: {result.skipped}
      </p>
    );
  }

  return (
    <p className="text-xs text-muted-foreground">
      Comments: {result.createdComments ?? 0}, skipped: {result.skipped}
    </p>
  );
}

function TestResult({
  result,
}: {
  result: AiContentAutomationTestResult | AiCommentAutomationTestResult | null;
}) {
  if (!result) return null;

  if (result.job === "content") {
    return (
      <div className="space-y-2 text-xs text-muted-foreground">
        <p>
          Test model: {result.provider} / {result.model}
        </p>
        <p>Category: {result.category.name}</p>
        <p>Title: {result.preview.title}</p>
        <p>Excerpt: {result.preview.excerpt}</p>
        <p>Body: {result.preview.bodyText}</p>
        <p>Tags: {result.preview.relatedTags.join(", ") || "None"}</p>
        <p>Image queries: {result.preview.imageSearchQueries.join(", ") || "None"}</p>
        {result.warnings.length > 0 ? <p className="text-amber-600">{result.warnings[0]}</p> : null}
      </div>
    );
  }

  return (
    <div className="space-y-2 text-xs text-muted-foreground">
      <p>
        Test model: {result.provider} / {result.model}
      </p>
      <p>Post: {result.post.title}</p>
      <p>Mode: {result.replyMode}</p>
      <p>Preview: {result.preview}</p>
    </div>
  );
}

export function AiAutomationSettingsScreen() {
  const [settings, setSettings] = useState<AiAutomationSettings | null>(null);
  const [categories, setCategories] = useState<CategoryItem[]>([]);
  const [activeTab, setActiveTab] = useState<SettingsTab>("ai");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [running, setRunning] = useState<"content" | "comments" | null>(null);
  const [testing, setTesting] = useState<"content" | "comments" | null>(null);
  const [lastRun, setLastRun] = useState<{ content: AiAutomationRunResult | null; comments: AiAutomationRunResult | null }>({
    content: null,
    comments: null,
  });
  const [lastTest, setLastTest] = useState<{
    content: AiContentAutomationTestResult | null;
    comments: AiCommentAutomationTestResult | null;
  }>({
    content: null,
    comments: null,
  });
  const [contentTestModalOpen, setContentTestModalOpen] = useState(false);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const [settingsData, categoryData] = await Promise.all([getAiAutomationSettings(), listAllCategories()]);
        setSettings(settingsData);
        setCategories(categoryData);
      } catch (error) {
        toast({
          title: "Failed to load AI automation settings",
          description: error instanceof Error ? error.message : undefined,
          variant: "error",
        });
      } finally {
        setLoading(false);
      }
    };

    void load();
  }, []);

  const categoryOptions = useMemo(
    () =>
      categories
        .filter((category) => Boolean(category.publishedAt))
        .map((category) => ({
          value: category.documentId,
          label: category.name,
        })),
    [categories],
  );

  const hasUnpublishedSelectedCategories = useMemo(() => {
    if (!settings) return false;
    const publishedIds = new Set(
      categories.filter((category) => Boolean(category.publishedAt)).map((category) => category.documentId),
    );
    return settings.content.categoryDocumentIds.some((documentId) => !publishedIds.has(documentId));
  }, [categories, settings]);

  const updateContent = (patch: Partial<AiAutomationSettings["content"]>) => {
    setSettings((current) => (current ? { ...current, content: { ...current.content, ...patch } } : current));
  };

  const updateComments = (patch: Partial<AiAutomationSettings["comments"]>) => {
    setSettings((current) => (current ? { ...current, comments: { ...current.comments, ...patch } } : current));
  };

  const updateProvider = (
    provider: "openai" | "anthropic",
    patch: Partial<AiAutomationSettings["providers"]["openai"]>,
  ) => {
    setSettings((current) =>
      current
        ? {
            ...current,
            providers: {
              ...current.providers,
              [provider]: {
                ...current.providers[provider],
                ...patch,
              },
            },
          }
        : current,
    );
  };

  const updateImageSearch = (
    provider: "google" | "pexels",
    patch: Partial<AiAutomationSettings["imageSearch"]["google"]>,
  ) => {
    setSettings((current) =>
      current
        ? {
            ...current,
            imageSearch: {
              ...current.imageSearch,
              [provider]: {
                ...current.imageSearch[provider],
                ...patch,
              },
            },
          }
        : current,
    );
  };

  const handleSave = async () => {
    if (!settings) return;
    setSaving(true);
    try {
      const saved = await updateAiAutomationSettings(settings);
      setSettings(saved);
      toast({ title: "AI automation settings saved", variant: "success" });
    } catch (error) {
      toast({
        title: "Failed to save settings",
        description: error instanceof Error ? error.message : undefined,
        variant: "error",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleRun = async (job: "content" | "comments") => {
    setRunning(job);
    try {
      const result = job === "content" ? await runAiContentCron() : await runAiCommentCron();
      const refreshed = await getAiAutomationSettings();
      setSettings(refreshed);
      setLastRun((current) => ({ ...current, [job]: result }));
      toast({
        title: job === "content" ? "AI content cron finished" : "AI comment cron finished",
        description: result.errors[0],
        variant: result.errors.length > 0 ? "error" : "success",
      });
    } catch (error) {
      toast({
        title: "Failed to run cron",
        description: error instanceof Error ? error.message : undefined,
        variant: "error",
      });
    } finally {
      setRunning(null);
    }
  };

  const handleCheckProvider = async (provider: "openai" | "anthropic") => {
    if (!settings) return;
    const selectedModels = settings.providers[provider].models;
    const apiKey = settings.providers[provider].apiKey.trim();
    if (!apiKey) {
      toast({
        title: "API key is required",
        description: `Please enter ${provider} API key first.`,
        variant: "error",
      });
      return;
    }

    try {
      const result = await checkAiProviderConnection({
        provider,
        apiKey,
        model: selectedModels[0],
      });
      toast({
        title: `${provider} connected`,
        description: result.message,
        variant: "success",
      });
    } catch (error) {
      toast({
        title: `Failed to connect ${provider}`,
        description: error instanceof Error ? error.message : undefined,
        variant: "error",
      });
    }
  };

  const handleTest = async (job: "content" | "comments") => {
    setTesting(job);
    try {
      const result = job === "content" ? await testAiContent() : await testAiComment();
      setLastTest((current) => ({ ...current, [job]: result }));
      if (job === "content") {
        setContentTestModalOpen(true);
      }
      toast({
        title: job === "content" ? "AI content test finished" : "AI comment test finished",
        description: `${result.provider} / ${result.model}`,
        variant: "success",
      });
    } catch (error) {
      toast({
        title: "Failed to test AI",
        description: error instanceof Error ? error.message : undefined,
        variant: "error",
      });
    } finally {
      setTesting(null);
    }
  };

  if (loading || !settings) {
    return <p className="text-sm text-muted-foreground">Loading AI automation settings...</p>;
  }

  return (
    <div className="space-y-6">
      {contentTestModalOpen && lastTest.content ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/55 p-4">
          <div className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-xl border bg-background shadow-xl">
            <div className="sticky top-0 flex items-center justify-between border-b bg-background px-5 py-4">
              <div>
                <h2 className="text-lg font-semibold">Content Test Result</h2>
                <p className="text-sm text-muted-foreground">
                  {lastTest.content.provider} / {lastTest.content.model} · {lastTest.content.category.name}
                </p>
              </div>
              <Button variant="ghost" size="icon-sm" onClick={() => setContentTestModalOpen(false)}>
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="space-y-6 p-5">
              <div className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
                <div className="space-y-3">
                  <div className="rounded-lg border p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Title</p>
                    <p className="mt-2 font-medium">{lastTest.content.preview.title}</p>
                  </div>
                  <div className="rounded-lg border p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Excerpt</p>
                    <p className="mt-2 text-sm text-muted-foreground">{lastTest.content.preview.excerpt}</p>
                  </div>
                  <div className="rounded-lg border p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Body</p>
                    <p className="mt-2 whitespace-pre-wrap text-sm">{lastTest.content.preview.bodyText}</p>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="rounded-lg border p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Tags</p>
                    <p className="mt-2 text-sm">{lastTest.content.preview.relatedTags.join(", ") || "None"}</p>
                  </div>
                  <div className="rounded-lg border p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">Image Queries</p>
                    <p className="mt-2 text-sm text-muted-foreground">
                      {lastTest.content.preview.imageSearchQueries.join(", ") || "None"}
                    </p>
                  </div>
                  {lastTest.content.warnings.length > 0 ? (
                    <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-700">
                      {lastTest.content.warnings[0]}
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium">Image Preview</h3>
                  <p className="text-sm text-muted-foreground">{lastTest.content.images.length} image(s)</p>
                </div>
                {lastTest.content.images.length > 0 ? (
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {lastTest.content.images.map((image, index) => (
                      <div key={`${image.url}-${index}`} className="overflow-hidden rounded-xl border bg-muted/20">
                        <div className="aspect-[4/3] overflow-hidden bg-muted">
                          <img
                            src={image.url}
                            alt={image.alt ?? `Preview ${index + 1}`}
                            className="h-full w-full object-cover"
                          />
                        </div>
                        <div className="space-y-1 p-3">
                          <p className="text-xs uppercase tracking-[0.2em] text-muted-foreground">{image.provider}</p>
                          <p className="line-clamp-2 text-sm">{image.alt ?? "No alt text"}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="rounded-lg border border-dashed p-6 text-sm text-muted-foreground">
                    No preview images returned from the configured image search providers.
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">AI Automation</h1>
          <p className="text-sm text-muted-foreground">
            Split by tabs so provider setup, image search, content, and comments stay separated.
          </p>
        </div>
        <Button onClick={() => void handleSave()} disabled={saving} className="gap-2">
          {saving ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
          {saving ? "Saving..." : "Save settings"}
        </Button>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          return (
            <Button
              key={tab.id}
              variant={activeTab === tab.id ? "default" : "outline"}
              className="justify-start gap-2"
              onClick={() => setActiveTab(tab.id)}
            >
              <Icon className="h-4 w-4" />
              {tab.label}
            </Button>
          );
        })}
      </div>

      {activeTab === "ai" ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bot className="h-4 w-4" />
              AI Providers
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {(["openai", "anthropic"] as const).map((provider) => (
              <div key={provider} className="space-y-4 rounded-lg border p-4">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="font-medium capitalize">{provider}</p>
                    <p className="text-xs text-muted-foreground">Used for content/comment generation.</p>
                  </div>
                  <Button variant="outline" onClick={() => void handleCheckProvider(provider)} className="gap-2">
                    <RefreshCw className="h-4 w-4" />
                    Check key
                  </Button>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Enabled</Label>
                    <Select
                      value={settings.providers[provider].enabled ? "true" : "false"}
                      onValueChange={(value) => updateProvider(provider, { enabled: value === "true" })}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="true">Enabled</SelectItem>
                        <SelectItem value="false">Disabled</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor={`${provider}-api-key`}>API key</Label>
                    <Input
                      id={`${provider}-api-key`}
                      type="password"
                      value={settings.providers[provider].apiKey}
                      onChange={(event) => updateProvider(provider, { apiKey: event.target.value })}
                      placeholder={`Enter ${provider} API key`}
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Models</Label>
                  <MultiSelectBox
                    options={[...PROVIDER_MODELS[provider]]}
                    value={settings.providers[provider].models}
                    onChange={(value) => updateProvider(provider, { models: value })}
                    placeholder={`Select ${provider} models`}
                  />
                  <p className="text-xs text-muted-foreground">
                    Content generation will randomly pick one model from all enabled models across providers.
                  </p>
                </div>

              </div>
            ))}
          </CardContent>
        </Card>
      ) : null}

      {activeTab === "images" ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ImageIcon className="h-4 w-4" />
              Image Search Providers
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="rounded-lg border bg-muted/30 p-3 text-sm text-muted-foreground">
              `auto` will try providers in this order: Google, Pexels. Settings values override `.env`, but
              `.env` still works as fallback.
            </div>

            {(["google", "pexels"] as const).map((provider) => (
              <div key={provider} className="space-y-4 rounded-lg border p-4">
                <div>
                  <p className="font-medium capitalize">{provider}</p>
                  <p className="text-xs text-muted-foreground">
                    {provider === "google"
                      ? "Google Custom Search JSON API for simple image discovery."
                      : "Free stock images from Pexels."}
                  </p>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <Label>Enabled</Label>
                    <Select
                      value={settings.imageSearch[provider].enabled ? "true" : "false"}
                      onValueChange={(value) => updateImageSearch(provider, { enabled: value === "true" })}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="true">Enabled</SelectItem>
                        <SelectItem value="false">Disabled</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor={`${provider}-search-api-key`}>API key</Label>
                    <Input
                      id={`${provider}-search-api-key`}
                      type="password"
                      value={settings.imageSearch[provider].apiKey}
                      onChange={(event) => updateImageSearch(provider, { apiKey: event.target.value })}
                      placeholder={`Enter ${provider} API key`}
                    />
                  </div>
                </div>

                {provider === "google" ? (
                  <div className="space-y-2">
                    <Label htmlFor="google-search-engine-id">Search engine ID (cx)</Label>
                    <Input
                      id="google-search-engine-id"
                      value={settings.imageSearch.google.searchEngineId ?? ""}
                      onChange={(event) => updateImageSearch("google", { searchEngineId: event.target.value })}
                      placeholder="Enter Google programmable search engine ID"
                    />
                  </div>
                ) : null}
              </div>
            ))}
          </CardContent>
        </Card>
      ) : null}

      {activeTab === "content" ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              Content Cron
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Enabled</Label>
                <Select
                  value={settings.content.enabled ? "true" : "false"}
                  onValueChange={(value) => updateContent({ enabled: value === "true" })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="true">Enabled</SelectItem>
                    <SelectItem value="false">Disabled</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="content-cron">Cron expression</Label>
                <Input
                  id="content-cron"
                  value={settings.content.cron}
                  onChange={(event) => updateContent({ cron: event.target.value })}
                />
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="content-posts">Posts per run</Label>
                <Input
                  id="content-posts"
                  type="number"
                  min={1}
                  max={10}
                  value={settings.content.postsPerRun}
                  onChange={(event) => updateContent({ postsPerRun: Number(event.target.value) || 1 })}
                />
              </div>
              <div className="space-y-2">
                <Label>Image provider</Label>
                <Select
                  value={settings.content.imageProvider}
                  onValueChange={(value) =>
                    updateContent({ imageProvider: value as AiAutomationSettings["content"]["imageProvider"] })
                  }
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto (Google, Pexels)</SelectItem>
                    <SelectItem value="google">Google</SelectItem>
                    <SelectItem value="pexels">Pexels</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="content-min-images">Min images</Label>
                <Input
                  id="content-min-images"
                  type="number"
                  min={1}
                  max={10}
                  value={settings.content.imageCountMin}
                  onChange={(event) => updateContent({ imageCountMin: Number(event.target.value) || 1 })}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="content-max-images">Max images</Label>
                <Input
                  id="content-max-images"
                  type="number"
                  min={1}
                  max={10}
                  value={settings.content.imageCountMax}
                  onChange={(event) => updateContent({ imageCountMax: Number(event.target.value) || 1 })}
                />
              </div>
              <div className="space-y-2">
                <Label>Preferred media mode</Label>
                <Select
                  value={settings.content.preferredMediaMode}
                  onValueChange={(value) =>
                    updateContent({ preferredMediaMode: value as AiAutomationSettings["content"]["preferredMediaMode"] })
                  }
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="body">Body images</SelectItem>
                    <SelectItem value="gallery">Gallery fallback</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Category filter</Label>
              <MultiSelectBox
                options={categoryOptions}
                value={settings.content.categoryDocumentIds}
                onChange={(value) => updateContent({ categoryDocumentIds: value })}
                placeholder="Leave empty to random all published categories"
              />
              <p className="text-xs text-muted-foreground">
                Optional. Leave empty to random from all published categories, or select a subset to random within it.
              </p>
              {hasUnpublishedSelectedCategories ? (
                <p className="text-xs text-destructive">
                  Current selection contains draft/unpublished categories. Re-select published categories, then save.
                </p>
              ) : null}
            </div>

            <div className="space-y-2">
              <Label htmlFor="content-prompt">Content prompt</Label>
              <Textarea
                id="content-prompt"
                rows={5}
                value={settings.content.contentPrompt}
                onChange={(event) => updateContent({ contentPrompt: event.target.value })}
              />
            </div>

            <div className="rounded-lg border bg-muted/30 p-3 text-sm">
              <p>Last run: {formatDate(settings.content.lastRunAt)}</p>
              <p>Last success: {formatDate(settings.content.lastSuccessAt)}</p>
              <p className={settings.content.lastError ? "text-destructive" : "text-muted-foreground"}>
                Last error: {settings.content.lastError ?? "None"}
              </p>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-3">
                <RunResult result={lastRun.content} />
                {lastTest.content ? (
                  <p className="text-xs text-muted-foreground">
                    Latest test: {lastTest.content.images.length} image(s), {lastTest.content.preview.relatedTags.length} tag(s)
                  </p>
                ) : null}
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <Button
                  variant="secondary"
                  onClick={() => void handleTest("content")}
                  disabled={running !== null || testing !== null}
                  className="gap-2"
                >
                  {testing === "content" ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
                  {testing === "content" ? "Testing..." : "Test content"}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => void handleRun("content")}
                  disabled={running !== null || testing !== null}
                  className="gap-2"
                >
                  {running === "content" ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Bot className="h-4 w-4" />}
                  {running === "content" ? "Running..." : "Run content now"}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : null}

      {activeTab === "comments" ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MessageSquareText className="h-4 w-4" />
              Comment Cron
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label>Enabled</Label>
                <Select
                  value={settings.comments.enabled ? "true" : "false"}
                  onValueChange={(value) => updateComments({ enabled: value === "true" })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="true">Enabled</SelectItem>
                    <SelectItem value="false">Disabled</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="comment-cron">Cron expression</Label>
                <Input
                  id="comment-cron"
                  value={settings.comments.cron}
                  onChange={(event) => updateComments({ cron: event.target.value })}
                />
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="comments-per-run">Comments per run</Label>
                <Input
                  id="comments-per-run"
                  type="number"
                  min={1}
                  max={20}
                  value={settings.comments.commentsPerRun}
                  onChange={(event) => updateComments({ commentsPerRun: Number(event.target.value) || 1 })}
                />
              </div>
              <div className="space-y-2">
                <Label>Allow replies</Label>
                <Select
                  value={settings.comments.allowReplies ? "true" : "false"}
                  onValueChange={(value) => updateComments({ allowReplies: value === "true" })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="true">Reply + top-level</SelectItem>
                    <SelectItem value="false">Top-level only</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="comment-prompt">Comment prompt</Label>
              <Textarea
                id="comment-prompt"
                rows={4}
                value={settings.comments.commentPrompt}
                onChange={(event) => updateComments({ commentPrompt: event.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="reply-prompt">Reply prompt</Label>
              <Textarea
                id="reply-prompt"
                rows={4}
                value={settings.comments.replyPrompt}
                onChange={(event) => updateComments({ replyPrompt: event.target.value })}
              />
            </div>

            <div className="rounded-lg border bg-muted/30 p-3 text-sm">
              <p>Scope: all published posts</p>
              <p>Actor source: random seeded user</p>
              <p>Last run: {formatDate(settings.comments.lastRunAt)}</p>
              <p>Last success: {formatDate(settings.comments.lastSuccessAt)}</p>
              <p className={settings.comments.lastError ? "text-destructive" : "text-muted-foreground"}>
                Last error: {settings.comments.lastError ?? "None"}
              </p>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="space-y-3">
                <RunResult result={lastRun.comments} />
                <TestResult result={lastTest.comments} />
              </div>
              <div className="flex flex-col gap-2 sm:flex-row">
                <Button
                  variant="secondary"
                  onClick={() => void handleTest("comments")}
                  disabled={running !== null || testing !== null}
                  className="gap-2"
                >
                  {testing === "comments" ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <Sparkles className="h-4 w-4" />
                  )}
                  {testing === "comments" ? "Testing..." : "Test comments"}
                </Button>
                <Button
                  variant="outline"
                  onClick={() => void handleRun("comments")}
                  disabled={running !== null || testing !== null}
                  className="gap-2"
                >
                  {running === "comments" ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <MessageSquareText className="h-4 w-4" />
                  )}
                  {running === "comments" ? "Running..." : "Run comments now"}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
