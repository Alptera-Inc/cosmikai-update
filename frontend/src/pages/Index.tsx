import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Database, Cpu, LineChart, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

const features = [
  {
    icon: Database,
    title: "Data Ingestion",
    description: "Import light curves from Kepler, K2, and TESS archives. FITS, CSV, and TSV formats with automated preprocessing pipelines.",
  },
  {
    icon: Cpu,
    title: "ML Inference",
    description: "Random Forest classifier trained on confirmed exoplanet transits. Configurable detection thresholds and feature extraction.",
  },
  {
    icon: LineChart,
    title: "3D Visualization",
    description: "Interactive orbital mechanics viewer with phase-folded light curves and planetary parameter estimation.",
  },
];

const Index = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border px-6 py-4">
        <div className="flex items-center gap-3">
          <img
            src="/cosmik-ai.png"
            alt="cosmik.ai"
            className="h-6 w-6"
          />
          <span className="font-data text-sm font-semibold tracking-wide">cosmik.ai</span>
        </div>
      </header>

      {/* Hero */}
      <section className="px-6 py-16 md:py-24">
        <div className="mx-auto max-w-4xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <img
              src="/cosmik-ai.png"
              alt="cosmik.ai"
              className="h-56 w-56 mb-0"
            />
            <p className="font-data text-xs text-muted-foreground uppercase tracking-widest mb-6">
              Astrophysics Data Pipeline v2.4.1
            </p>
            <h1 className="text-3xl md:text-4xl font-semibold leading-tight tracking-tight mb-6 mt-2">
              Cosmik AI: Stellar Photometry & Transit Analysis
            </h1>
            <div className="max-w-3xl text-sm text-muted-foreground leading-relaxed space-y-4 mb-10">
              <p>
                A machine learning pipeline for detecting exoplanetary transit signals in stellar photometry data. 
                The system employs ensemble Random Forest classifiers trained on labeled Kepler Objects of Interest (KOI) 
                to identify periodic flux variations consistent with planetary transits.
              </p>
              <p>
                Supported missions: <span className="font-data text-foreground">Kepler</span>, <span className="font-data text-foreground">K2</span>, <span className="font-data text-foreground">TESS</span>. 
                Input formats: FITS, CSV, TSV. Feature extraction includes transit depth (δ), orbital period (P), 
                signal-to-noise ratio (SNR), and transit duration (T<sub>14</sub>).
              </p>
            </div>
            <div className="flex gap-3">
              <Button
                onClick={() => navigate("/dashboard")}
                className="bg-foreground text-background hover:bg-foreground/90 font-data text-xs uppercase tracking-wider px-6 rounded-md"
              >
                Initialize Workspace
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
              <Button
                variant="outline"
                onClick={() => navigate("/visualizer")}
                className="border-border text-muted-foreground hover:text-foreground hover:bg-card font-data text-xs uppercase tracking-wider px-6 rounded-md"
              >
                Open Visualizer
              </Button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="px-6 pb-16">
        <div className="mx-auto max-w-4xl">
          <div className="grid gap-4 md:grid-cols-3">
            {features.map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.4, delay: 0.2 + i * 0.1 }}
                className="panel p-6"
              >
                <feature.icon className="h-5 w-5 text-muted-foreground mb-4" />
                <h3 className="font-data text-xs uppercase tracking-wider text-foreground mb-2">{feature.title}</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border px-6 py-4">
        <div className="mx-auto max-w-4xl flex items-center justify-between">
          <p className="font-data text-xs text-muted-foreground">
            © 2024 Cosmik AI Research
          </p>
          <p className="font-data text-xs text-muted-foreground">
            Build: <span className="text-foreground">2024.03.15</span>
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
