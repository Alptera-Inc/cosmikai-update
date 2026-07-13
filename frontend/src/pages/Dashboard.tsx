import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Upload, Search, Sliders, BarChart3, Eye,
  FileUp, CheckCircle2, AlertTriangle, Target, Activity,
  Loader2, Radio, Clock, Zap, AlertCircle, ChevronDown, ChevronRight,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Checkbox } from "@/components/ui/checkbox";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PredictionResult {
  star_name: string;
  mission: string;
  score: number;
  percentage: number;
  period_days: number | null;
  verdict: string;
  transit_depth_estimate: number | null;
  duration_estimate: number | null;
  num_datapoints: number | null;
  cached: boolean;
  processing_time_seconds: number;
  timestamp: string;
}

interface PlanetResult {
  score: number;
  percentage: number;
  period_days: number | null;
  verdict: string;
  transit_depth_estimate: number | null;
}

interface MultiPlanetResult {
  star_name: string;
  mission: string;
  planets: PlanetResult[];
  num_datapoints: number | null;
  processing_time_seconds: number;
  timestamp: string;
  cached?: boolean;
}

interface HistoryItem {
  id: number;
  star_name: string;
  mission: string;
  score: number;
  percentage: number;
  period_days: number | null;
  verdict: string;
  num_datapoints: number | null;
  timestamp: string;
}

interface HistoryResponse {
  items: HistoryItem[];
  total: number;
}

interface StatsResponse {
  total_analyzed: number;
  average_score: number;
  detection_rate: number;
  missions: Record<string, number>;
  above_threshold: number;
  below_threshold: number;
}

interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_version: string;
  model_auprc: number;
  total_predictions: number;
  uptime_seconds: number;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

const fetchStats = async (): Promise<StatsResponse> => {
  const res = await fetch("/api/stats");
  if (!res.ok) throw new Error("Failed to load statistics");
  return res.json();
};

const fetchHistory = async (): Promise<HistoryResponse> => {
  const res = await fetch("/api/history?limit=50&sort=timestamp&order=desc");
  if (!res.ok) throw new Error("Failed to load history");
  return res.json();
};

const fetchHealth = async (): Promise<HealthResponse> => {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error("Failed to load health status");
  return res.json();
};

// ---------------------------------------------------------------------------
// Mission mapping — select value → API string
// ---------------------------------------------------------------------------

const MISSION_MAP: Record<string, string> = {
  kepler: "Kepler",
  k2: "K2",
  tess: "TESS",
};

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

const Dashboard = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Model config state
  const [threshold, setThreshold] = useState([50]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [features, setFeatures] = useState({
    blsPeriodSearch: true,
    phaseFold: true,
    fluxNormalization: true,
    transitDepthEstimate: true,
  });

  // Query state
  const [starQuery, setStarQuery] = useState("");
  const [selectedMission, setSelectedMission] = useState("kepler");

  // Prediction state
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResult | MultiPlanetResult | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  // History expand state
  const [expandedStars, setExpandedStars] = useState<Set<string>>(new Set());
  const toggleStar = (name: string) =>
    setExpandedStars((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });

  // Confidence colour helper
  const confidenceColor = (pct: number) =>
    pct >= 70 ? "text-success" : pct >= 40 ? "text-warning" : "text-destructive";
  const confidenceBg = (pct: number) =>
    pct >= 70 ? "bg-success" : pct >= 40 ? "bg-warning" : "bg-destructive";

  // Type guard
  const isMultiPlanet = (r: PredictionResult | MultiPlanetResult): r is MultiPlanetResult =>
    "planets" in r && Array.isArray((r as MultiPlanetResult).planets);

  // Remote data (stats, history, health)
  const { data: statsData } = useQuery<StatsResponse>({
    queryKey: ["stats"],
    queryFn: fetchStats,
    refetchInterval: 30_000,
    retry: 2,
  });

  const { data: historyData } = useQuery<HistoryResponse>({
    queryKey: ["history"],
    queryFn: fetchHistory,
    refetchInterval: 30_000,
    retry: 2,
  });

  const { data: healthData } = useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 60_000,
    retry: 2,
  });

  // ── Telemetry stats cards — real data with graceful fallback ──────────────
  const telemetryStats = [
    {
      label: "MODEL_ACCURACY",
      value: healthData ? `${(healthData.model_auprc * 100).toFixed(1)}%` : "—",
      sub: "AUPRC",
      icon: Target,
    },
    {
      label: "PROCESSED_TARGETS",
      value: statsData ? statsData.total_analyzed.toLocaleString() : "—",
      sub: "stars analysed",
      icon: Activity,
    },
    {
      label: "CANDIDATE_FLAGS",
      value: statsData ? statsData.above_threshold.toLocaleString() : "—",
      sub: `score ≥ ${threshold[0]}%`,
      icon: CheckCircle2,
    },
    {
      label: "FALSE_POSITIVES",
      value: statsData ? statsData.below_threshold.toLocaleString() : "—",
      sub: `score < ${threshold[0]}%`,
      icon: AlertTriangle,
    },
  ];

  // ── "Query Archive" handler ──────────────────────────────────────────────
  const handleQueryArchive = async () => {
    if (!starQuery.trim()) return;

    setIsLoading(true);
    setPredictionResult(null);
    setPredictionError(null);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          star_name: starQuery.trim(),
          mission: MISSION_MAP[selectedMission] ?? "Kepler",
          threshold: threshold[0] / 100,
        }),
        // 3-minute client-side timeout matches the server REQUEST_TIMEOUT
        signal: AbortSignal.timeout(195_000),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
        throw new Error(err.detail ?? `Server error: ${res.status}`);
      }

      const data: PredictionResult | MultiPlanetResult = await res.json();
      setPredictionResult(data);

      // Refresh history and stats after a new prediction
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["history"] }),
        queryClient.invalidateQueries({ queryKey: ["stats"] }),
      ]);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "TimeoutError") {
        setPredictionError(
          "Request timed out after 3 minutes. MAST may be slow — please try again."
        );
      } else if (err instanceof Error) {
        setPredictionError(err.message);
      } else {
        setPredictionError("An unexpected error occurred.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  // ── Model status indicator ────────────────────────────────────────────────
  const modelOnline = healthData?.model_loaded ?? false;

  return (
    <div className="min-h-screen bg-background">
      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <header className="border-b border-border px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            onClick={() => navigate("/")}
            className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
          >
            <img
              src="/cosmik-ai.png"
              alt="cosmik.ai"
              className="h-6 w-6"
            />
            <span className="font-data text-sm font-semibold tracking-wide">cosmik.ai</span>
          </div>
          <span className="font-data text-xs text-muted-foreground ml-4">/ DASHBOARD</span>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 font-data text-xs">
            <span
              className={`h-2 w-2 rounded-full ${modelOnline ? "bg-success" : "bg-destructive"}`}
            />
            <span className="text-muted-foreground">MODEL_STATUS:</span>
            <span className={modelOnline ? "text-success" : "text-destructive"}>
              {modelOnline ? "ACTIVE" : "OFFLINE"}
            </span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/visualizer")}
            className="font-data text-xs uppercase tracking-wider text-muted-foreground hover:text-foreground"
          >
            <Eye className="mr-2 h-4 w-4" /> Visualizer
          </Button>
        </div>
      </header>

      {/* ── Content ─────────────────────────────────────────────────────────── */}
      <main className="p-6">
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.3 }}>
          <Tabs defaultValue="input" className="space-y-6">
            <TabsList className="bg-card border border-border rounded-md p-1 h-auto">
              <TabsTrigger
                value="input"
                className="font-data text-xs uppercase tracking-wider data-[state=active]:bg-muted data-[state=active]:text-foreground rounded-md px-4 py-2"
              >
                <Upload className="mr-2 h-3 w-3" /> FITS/CSV Ingestion
              </TabsTrigger>
              <TabsTrigger
                value="config"
                className="font-data text-xs uppercase tracking-wider data-[state=active]:bg-muted data-[state=active]:text-foreground rounded-md px-4 py-2"
              >
                <Sliders className="mr-2 h-3 w-3" /> Model Parameters
              </TabsTrigger>
              <TabsTrigger
                value="stats"
                className="font-data text-xs uppercase tracking-wider data-[state=active]:bg-muted data-[state=active]:text-foreground rounded-md px-4 py-2"
              >
                <BarChart3 className="mr-2 h-3 w-3" /> System Telemetry
              </TabsTrigger>
            </TabsList>

            {/* ── FITS/CSV Ingestion Tab ─────────────────────────────────────── */}
            <TabsContent value="input">
              <div className="grid gap-6 lg:grid-cols-2">
                {/* File Upload */}
                <div
                  className={`panel p-6 flex flex-col items-center justify-center min-h-[280px] transition-colors cursor-pointer ${
                    isDragOver ? "border-foreground bg-muted" : "hover:border-muted-foreground"
                  }`}
                  onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                  onDragLeave={() => setIsDragOver(false)}
                  onDrop={(e) => { e.preventDefault(); setIsDragOver(false); }}
                >
                  <FileUp className="h-8 w-8 text-muted-foreground mb-4" />
                  <p className="font-data text-xs uppercase tracking-wider mb-1">
                    Drop Light Curve Files
                  </p>
                  <p className="text-xs text-muted-foreground mb-4">FITS, CSV, TSV formats accepted</p>
                  <Button
                    variant="outline"
                    size="sm"
                    className="font-data text-xs uppercase tracking-wider border-border hover:bg-muted rounded-md"
                  >
                    Browse Files
                  </Button>
                </div>

                {/* Query by Star */}
                <div className="panel p-6 space-y-6">
                  <div>
                    <p className="font-data text-xs uppercase tracking-wider text-foreground mb-1">
                      Query by Star ID
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Download photometry from MAST and run transit detection
                    </p>
                  </div>
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <Label className="font-data text-xs uppercase tracking-wider text-muted-foreground">
                        Star Identifier
                      </Label>
                      <div className="relative">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
                        <Input
                          value={starQuery}
                          onChange={(e) => setStarQuery(e.target.value)}
                          onKeyDown={(e) => e.key === "Enter" && !isLoading && handleQueryArchive()}
                          placeholder="e.g. Kepler-10, TOI-700, KIC 11904151"
                          disabled={isLoading}
                          className="pl-9 bg-background border-border font-data text-sm focus-visible:ring-1 focus-visible:ring-foreground rounded-md"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label className="font-data text-xs uppercase tracking-wider text-muted-foreground">
                        Mission
                      </Label>
                      <Select
                        value={selectedMission}
                        onValueChange={setSelectedMission}
                        disabled={isLoading}
                      >
                        <SelectTrigger className="bg-background border-border font-data text-sm rounded-md">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-card border-border rounded-md">
                          <SelectItem value="kepler" className="font-data text-sm">Kepler</SelectItem>
                          <SelectItem value="k2" className="font-data text-sm">K2</SelectItem>
                          <SelectItem value="tess" className="font-data text-sm">TESS</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <Button
                      className="w-full bg-foreground text-background hover:bg-foreground/90 font-data text-xs uppercase tracking-wider rounded-md"
                      onClick={handleQueryArchive}
                      disabled={isLoading || !starQuery.trim()}
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                          Downloading from MAST…
                        </>
                      ) : (
                        <>
                          <Search className="mr-2 h-3 w-3" /> Query Archive
                        </>
                      )}
                    </Button>

                    {isLoading && (
                      <p className="font-data text-[10px] text-muted-foreground text-center">
                        MAST downloads take 30–90 s for new stars. Cached results return instantly.
                      </p>
                    )}
                  </div>
                </div>
              </div>

              {/* ── Error panel ──────────────────────────────────────────────── */}
              {predictionError && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="panel p-5 border-destructive/50 bg-destructive/5 flex items-start gap-3 mt-2"
                >
                  <AlertCircle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-data text-xs uppercase tracking-wider text-destructive mb-1">
                      Detection Failed
                    </p>
                    <p className="text-xs text-muted-foreground">{predictionError}</p>
                  </div>
                </motion.div>
              )}

              {/* ── Result panel ─────────────────────────────────────────────── */}
              {predictionResult && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="panel p-6 space-y-5 mt-2"
                >
                  {isMultiPlanet(predictionResult) ? (
                    <>
                      {/* Star System Header */}
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-data text-xs uppercase tracking-wider text-muted-foreground mb-1">
                            Detection Result
                          </p>
                          <p className="font-data text-sm font-semibold">
                            {predictionResult.star_name}
                            <span className="ml-2 text-muted-foreground font-normal">
                              / {predictionResult.mission}
                            </span>
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          {predictionResult.cached && (
                            <span className="bg-muted px-2 py-0.5 rounded-md font-data text-[10px] uppercase tracking-wider text-muted-foreground">
                              CACHED
                            </span>
                          )}
                          <span className="bg-muted px-2 py-0.5 rounded-md font-data text-[10px] uppercase tracking-wider text-muted-foreground">
                            {predictionResult.planets.length} planet{predictionResult.planets.length !== 1 ? "s" : ""} detected
                          </span>
                        </div>
                      </div>

                      {/* Planet Cards */}
                      <div className="space-y-3">
                        {predictionResult.planets.map((planet, i) => (
                          <div key={i} className="bg-muted rounded-md p-4 space-y-3 border border-border">
                            <div className="flex items-center justify-between">
                              <p className="font-data text-xs uppercase tracking-wider">
                                Planet Candidate {i + 1}
                              </p>
                              <span
                                className={`px-2 py-0.5 rounded-md font-data text-[10px] uppercase tracking-wider ${
                                  planet.verdict === "TRANSIT_DETECTED"
                                    ? "bg-success/15 text-success"
                                    : "bg-warning/15 text-warning"
                                }`}
                              >
                                {planet.verdict.replace("_", " ")}
                              </span>
                            </div>
                            <div className="flex items-end gap-3">
                              <span className={`font-data text-3xl font-semibold ${confidenceColor(planet.percentage)}`}>
                                {planet.percentage.toFixed(1)}%
                              </span>
                              <span className="font-data text-xs text-muted-foreground mb-1">transit confidence</span>
                            </div>
                            <div className="w-full bg-background rounded-full h-1">
                              <div
                                className={`h-1 rounded-full transition-all duration-700 ${confidenceBg(planet.percentage)}`}
                                style={{ width: `${planet.percentage}%` }}
                              />
                            </div>
                            <div className="grid grid-cols-2 gap-2">
                              <div className="bg-background rounded-md p-2 border border-border">
                                <div className="flex items-center gap-1 mb-1">
                                  <Radio className="h-3 w-3 text-muted-foreground" />
                                  <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">Period</p>
                                </div>
                                <p className="font-data text-sm font-semibold">
                                  {planet.period_days != null ? `${planet.period_days.toFixed(3)} d` : "—"}
                                </p>
                              </div>
                              <div className="bg-background rounded-md p-2 border border-border">
                                <div className="flex items-center gap-1 mb-1">
                                  <Zap className="h-3 w-3 text-muted-foreground" />
                                  <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">Transit Depth</p>
                                </div>
                                <p className="font-data text-sm font-semibold">
                                  {planet.transit_depth_estimate != null
                                    ? `${(planet.transit_depth_estimate * 100).toFixed(4)}%`
                                    : "—"}
                                </p>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Footer metrics */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-muted p-3 rounded-md">
                          <div className="flex items-center gap-1 mb-1">
                            <Activity className="h-3 w-3 text-muted-foreground" />
                            <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">Datapoints</p>
                          </div>
                          <p className="font-data text-sm font-semibold">
                            {predictionResult.num_datapoints != null
                              ? predictionResult.num_datapoints.toLocaleString()
                              : "—"}
                          </p>
                        </div>
                        <div className="bg-muted p-3 rounded-md">
                          <div className="flex items-center gap-1 mb-1">
                            <Clock className="h-3 w-3 text-muted-foreground" />
                            <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">Proc. Time</p>
                          </div>
                          <p className="font-data text-sm font-semibold">
                            {predictionResult.processing_time_seconds.toFixed(1)}s
                          </p>
                        </div>
                      </div>
                      <p className="font-data text-[10px] text-muted-foreground">
                        {new Date(predictionResult.timestamp).toLocaleString()} UTC
                      </p>
                    </>
                  ) : (
                    <>
                      {/* Header row */}
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-data text-xs uppercase tracking-wider text-muted-foreground mb-1">
                            Detection Result
                          </p>
                          <p className="font-data text-sm font-semibold">
                            {predictionResult.star_name}
                            <span className="ml-2 text-muted-foreground font-normal">
                              / {predictionResult.mission}
                            </span>
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          {predictionResult.cached && (
                            <span className="bg-muted px-2 py-0.5 rounded-md font-data text-[10px] uppercase tracking-wider text-muted-foreground">
                              CACHED
                            </span>
                          )}
                          <span
                            className={`px-2 py-1 rounded-md font-data text-[10px] uppercase tracking-wider ${
                              predictionResult.verdict === "TRANSIT_DETECTED"
                                ? "bg-success/15 text-success"
                                : "bg-warning/15 text-warning"
                            }`}
                          >
                            {predictionResult.verdict.replace("_", " ")}
                          </span>
                        </div>
                      </div>

                      {/* Score */}
                      <div className="flex items-end gap-3">
                        <span
                          className={`font-data text-4xl font-semibold ${
                            predictionResult.verdict === "TRANSIT_DETECTED" ? "text-success" : "text-warning"
                          }`}
                        >
                          {predictionResult.percentage.toFixed(1)}%
                        </span>
                        <span className="font-data text-xs text-muted-foreground mb-1">
                          transit confidence
                        </span>
                      </div>

                      {/* Score bar */}
                      <div className="w-full bg-muted rounded-full h-1.5">
                        <div
                          className={`h-1.5 rounded-full transition-all duration-700 ${
                            predictionResult.verdict === "TRANSIT_DETECTED" ? "bg-success" : "bg-warning"
                          }`}
                          style={{ width: `${predictionResult.percentage}%` }}
                        />
                      </div>

                      {/* Metrics grid */}
                      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                        {[
                          {
                            icon: Radio,
                            label: "Period",
                            value: predictionResult.period_days != null
                              ? `${predictionResult.period_days.toFixed(3)} d`
                              : "—",
                          },
                          {
                            icon: Zap,
                            label: "Transit Depth",
                            value: predictionResult.transit_depth_estimate != null
                              ? `${(predictionResult.transit_depth_estimate * 100).toFixed(4)}%`
                              : "—",
                          },
                          {
                            icon: Activity,
                            label: "Datapoints",
                            value: predictionResult.num_datapoints != null
                              ? predictionResult.num_datapoints.toLocaleString()
                              : "—",
                          },
                          {
                            icon: Clock,
                            label: "Proc. Time",
                            value: `${predictionResult.processing_time_seconds.toFixed(1)}s`,
                          },
                        ].map((m) => (
                          <div key={m.label} className="bg-muted p-3 rounded-md">
                            <div className="flex items-center gap-1 mb-1">
                              <m.icon className="h-3 w-3 text-muted-foreground" />
                              <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">
                                {m.label}
                              </p>
                            </div>
                            <p className="font-data text-sm font-semibold">{m.value}</p>
                          </div>
                        ))}
                      </div>

                      <p className="font-data text-[10px] text-muted-foreground">
                        {new Date(predictionResult.timestamp).toLocaleString()} UTC
                      </p>
                    </>
                  )}
                </motion.div>
              )}
            </TabsContent>

            {/* ── Model Parameters Tab ──────────────────────────────────────── */}
            <TabsContent value="config">
              <div className="grid gap-6 lg:grid-cols-2">
                {/* Threshold + model type */}
                <div className="panel p-6 space-y-6">
                  <div>
                    <p className="font-data text-xs uppercase tracking-wider text-foreground mb-1">
                      Detection Threshold
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Verdict cutoff applied to the CNN sigmoid score.
                      Predictions below this value are classified as NO_TRANSIT.
                    </p>
                  </div>
                  <div className="space-y-4">
                    <div className="flex justify-between font-data text-xs">
                      <span className="text-muted-foreground">Conservative</span>
                      <span className="text-foreground font-semibold">{threshold[0]}%</span>
                      <span className="text-muted-foreground">Aggressive</span>
                    </div>
                    <Slider
                      value={threshold}
                      onValueChange={setThreshold}
                      min={30}
                      max={99}
                      step={1}
                      className="[&_[role=slider]]:bg-foreground [&_[role=slider]]:border-foreground"
                    />
                    <p className="font-data text-[10px] text-muted-foreground">
                      Current: score ≥ {threshold[0]}% → TRANSIT_DETECTED.
                      Applied to new queries and re-evaluates the telemetry counts above.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label className="font-data text-xs uppercase tracking-wider text-muted-foreground">
                      Deployed Model
                    </Label>
                    <div className="w-full bg-background border border-border rounded-md px-3 py-2 flex items-center justify-between">
                      <span className="font-data text-sm">1D Convolutional Neural Network</span>
                      <span className="font-data text-[10px] text-muted-foreground bg-muted px-2 py-0.5 rounded-md uppercase tracking-wider">
                        TransitCNN v{healthData?.model_version ?? "1.0.0"}
                      </span>
                    </div>
                    <p className="font-data text-[10px] text-muted-foreground">
                      88,961 parameters · 288 KB · optimised for satellite edge deployment
                    </p>
                  </div>
                </div>

                {/* Feature extraction */}
                <div className="panel p-6 space-y-6">
                  <div>
                    <p className="font-data text-xs uppercase tracking-wider text-foreground mb-1">
                      Feature Extraction Pipeline
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Steps applied to every light curve before CNN inference.
                      All steps are required for the trained model.
                    </p>
                  </div>
                  <div className="space-y-3">
                    {[
                      {
                        key: "blsPeriodSearch",
                        label: "BLS_PERIOD_SEARCH",
                        desc: "Box Least Squares · 5000 trial periods · 0.6–12.0 d",
                      },
                      {
                        key: "phaseFold",
                        label: "PHASE_FOLD + BIN_512",
                        desc: "Fold to best period · median-bin to 512 points",
                      },
                      {
                        key: "fluxNormalization",
                        label: "FLUX_NORMALISATION",
                        desc: "Savitzky-Golay detrend · median normalise · (x−μ)/σ standardise",
                      },
                      {
                        key: "transitDepthEstimate",
                        label: "TRANSIT_DEPTH_ESTIMATE",
                        desc: "Fractional depth δ from BLS best-fit box model",
                      },
                    ].map((f) => (
                      <label
                        key={f.key}
                        className="flex items-start gap-3 p-3 bg-background border border-border rounded-md cursor-pointer transition-colors hover:border-muted-foreground"
                      >
                        <Checkbox
                          checked={features[f.key as keyof typeof features]}
                          onCheckedChange={(checked) =>
                            setFeatures((p) => ({ ...p, [f.key]: !!checked }))
                          }
                          className="mt-0.5 border-border data-[state=checked]:bg-foreground data-[state=checked]:border-foreground rounded-sm"
                        />
                        <div>
                          <span className="font-data text-xs tracking-wide block">{f.label}</span>
                          <span className="font-data text-[10px] text-muted-foreground">{f.desc}</span>
                        </div>
                      </label>
                    ))}
                  </div>
                  <Button className="w-full bg-foreground text-background hover:bg-foreground/90 font-data text-xs uppercase tracking-wider rounded-md">
                    Apply Configuration
                  </Button>
                </div>
              </div>
            </TabsContent>

            {/* ── System Telemetry Tab ──────────────────────────────────────── */}
            <TabsContent value="stats">
              <div className="space-y-6">
                {/* Stats cards */}
                <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
                  {telemetryStats.map((stat, i) => (
                    <motion.div
                      key={stat.label}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.05 }}
                      className="panel p-4"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-data text-[10px] text-muted-foreground uppercase tracking-wider">
                          {stat.label}
                        </span>
                        <stat.icon className="h-3 w-3 text-muted-foreground" />
                      </div>
                      <p className="font-data text-2xl font-semibold">{stat.value}</p>
                      <p className="font-data text-[10px] text-muted-foreground mt-1">{stat.sub}</p>
                    </motion.div>
                  ))}
                </div>

                {/* Mission breakdown */}
                {statsData && Object.keys(statsData.missions).length > 0 && (
                  <div className="panel p-4">
                    <p className="font-data text-xs uppercase tracking-wider text-foreground mb-3">
                      Mission Breakdown
                    </p>
                    <div className="flex gap-4">
                      {Object.entries(statsData.missions).map(([mission, count]) => (
                        <div key={mission} className="bg-muted rounded-md px-4 py-2 text-center">
                          <p className="font-data text-xs text-muted-foreground uppercase tracking-wider">
                            {mission}
                          </p>
                          <p className="font-data text-xl font-semibold">{count}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Detection History */}
                <div className="panel p-6">
                  <div className="flex items-center justify-between mb-4">
                    <p className="font-data text-xs uppercase tracking-wider text-foreground">
                      Detection History
                    </p>
                    {historyData && (
                      <span className="font-data text-[10px] text-muted-foreground">
                        {historyData.total} total record{historyData.total !== 1 ? "s" : ""}
                      </span>
                    )}
                  </div>

                  {!historyData || historyData.items.length === 0 ? (
                    <div className="text-center py-12">
                      <Search className="h-6 w-6 text-muted-foreground mx-auto mb-3" />
                      <p className="font-data text-xs text-muted-foreground uppercase tracking-wider">
                        No predictions yet
                      </p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Query a star in the FITS/CSV Ingestion tab to get started.
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-1">
                      {Object.entries(
                        historyData.items.reduce((acc, item) => {
                          if (!acc[item.star_name])
                            acc[item.star_name] = { mission: item.mission, items: [] };
                          acc[item.star_name].items.push(item);
                          return acc;
                        }, {} as Record<string, { mission: string; items: HistoryItem[] }>)
                      ).map(([starName, group]) => {
                        const isExpanded = expandedStars.has(starName);
                        const hasTransit = group.items.some((i) => i.verdict === "TRANSIT_DETECTED");
                        return (
                          <div key={starName} className="border border-border rounded-md overflow-hidden">
                            {/* Star header row */}
                            <button
                              onClick={() => toggleStar(starName)}
                              className="w-full flex items-center gap-3 px-4 py-3 bg-muted/30 hover:bg-muted/60 transition-colors text-left"
                            >
                              {isExpanded
                                ? <ChevronDown className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                                : <ChevronRight className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                              }
                              <span className="font-data text-xs font-semibold flex-1">{starName}</span>
                              <span className="bg-muted px-2 py-0.5 rounded-md font-data text-[10px] uppercase tracking-wider text-muted-foreground mr-3">
                                {group.mission}
                              </span>
                              <span className="font-data text-[10px] text-muted-foreground mr-3">
                                {group.items.length} detection{group.items.length !== 1 ? "s" : ""}
                              </span>
                              <span
                                className={`font-data text-[10px] uppercase tracking-wider ${
                                  hasTransit ? "text-success" : "text-warning"
                                }`}
                              >
                                {hasTransit ? "TRANSIT" : "NO TRANSIT"}
                              </span>
                            </button>

                            {/* Planet rows */}
                            {isExpanded && (
                              <div className="border-t border-border">
                                {group.items.map((row, idx) => (
                                  <div
                                    key={row.id}
                                    className={`flex items-center gap-4 px-8 py-2.5 font-data text-xs ${
                                      idx < group.items.length - 1 ? "border-b border-border/40" : ""
                                    } hover:bg-muted/30 transition-colors`}
                                  >
                                    <span className="text-muted-foreground w-36 flex-shrink-0">
                                      Planet Candidate {idx + 1}
                                    </span>
                                    <span className="text-muted-foreground flex-1">
                                      {row.period_days != null ? `${row.period_days.toFixed(3)} d` : "—"}
                                    </span>
                                    <span
                                      className={
                                        row.percentage >= 70
                                          ? "text-success"
                                          : row.percentage >= 40
                                          ? "text-warning"
                                          : "text-destructive"
                                      }
                                    >
                                      {row.percentage.toFixed(1)}%
                                    </span>
                                    <span
                                      className={`text-[10px] uppercase tracking-wider w-32 text-right ${
                                        row.verdict === "TRANSIT_DETECTED" ? "text-success" : "text-warning"
                                      }`}
                                    >
                                      {row.verdict.replace("_", " ")}
                                    </span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                {/* Server health */}
                {healthData && (
                  <div className="panel p-4">
                    <p className="font-data text-xs uppercase tracking-wider text-foreground mb-3">
                      Server Status
                    </p>
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      {[
                        { label: "API Status", value: healthData.status.toUpperCase() },
                        {
                          label: "Model",
                          value: healthData.model_loaded ? "LOADED" : "OFFLINE",
                        },
                        {
                          label: "Uptime",
                          value: `${Math.floor(healthData.uptime_seconds / 60)}m`,
                        },
                        {
                          label: "AUPRC",
                          value: `${(healthData.model_auprc * 100).toFixed(2)}%`,
                        },
                      ].map((s) => (
                        <div key={s.label} className="bg-muted rounded-md p-3">
                          <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
                            {s.label}
                          </p>
                          <p className="font-data text-sm font-semibold">{s.value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </motion.div>
      </main>
    </div>
  );
};

export default Dashboard;
