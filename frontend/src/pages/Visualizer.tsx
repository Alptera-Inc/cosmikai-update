import { useState, useRef, Suspense } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Search, ArrowLeft } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import * as THREE from "three";

// Mock light curve data with transit dip
const lightCurveData = Array.from({ length: 100 }, (_, i) => {
  const phase = (i - 50) / 50;
  let flux = 1.0;
  if (Math.abs(phase) < 0.12) {
    flux = 1.0 - 0.008 * Math.cos((phase / 0.12) * Math.PI * 0.5) ** 2;
  }
  flux += (Math.random() - 0.5) * 0.001;
  return { phase: phase.toFixed(3), flux: parseFloat(flux.toFixed(5)) };
});

const planets = [
  {
    name: "Kepler-10 b",
    period: 0.837,
    sizeVsEarth: 1.47,
    semiMajorAxis: 0.0168,
    temp: 2169,
    probability: 94.7,
    orbitRadius: 3,
    size: 0.15,
    color: "#c97642",
    orbitSpeed: 2,
  },
  {
    name: "Kepler-10 c",
    period: 45.29,
    sizeVsEarth: 2.35,
    semiMajorAxis: 0.241,
    temp: 584,
    probability: 88.0,
    orbitRadius: 5.5,
    size: 0.24,
    color: "#4a7c9b",
    orbitSpeed: 0.4,
  },
];

function Star() {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.1;
    }
  });

  return (
    <group>
      <mesh ref={meshRef}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          color="#ffd54f"
          emissive="#ff8a00"
          emissiveIntensity={1.2}
        />
      </mesh>
      <pointLight position={[0, 0, 0]} intensity={2} color="#FDB813" distance={30} decay={2} />
    </group>
  );
}

function Planet({ orbitRadius, size, color, orbitSpeed, isSelected, onClick }: {
  orbitRadius: number;
  size: number;
  color: string;
  orbitSpeed: number;
  isSelected: boolean;
  onClick: () => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = state.clock.elapsedTime * orbitSpeed;
    }
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.5;
    }
  });

  return (
    <group ref={groupRef}>
      <mesh ref={meshRef} position={[orbitRadius, 0, 0]} onClick={onClick}>
        <sphereGeometry args={[size, 32, 32]} />
        <meshStandardMaterial color={color} roughness={0.7} metalness={0.3} />
        {isSelected && (
          <mesh>
            <ringGeometry args={[size + 0.08, size + 0.12, 32]} />
            <meshBasicMaterial color="#ffffff" side={THREE.DoubleSide} transparent opacity={0.8} />
          </mesh>
        )}
      </mesh>
    </group>
  );
}

function OrbitPath({ radius }: { radius: number }) {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]}>
      <ringGeometry args={[radius - 0.02, radius + 0.02, 128]} />
      <meshBasicMaterial color="#333333" side={THREE.DoubleSide} transparent opacity={0.4} />
    </mesh>
  );
}

function Scene({ selectedPlanet, setSelectedPlanet }: { selectedPlanet: number; setSelectedPlanet: (i: number) => void }) {
  return (
    <>
      <ambientLight intensity={0.5} />
      <Stars radius={100} depth={50} count={3000} factor={4} saturation={0} fade speed={1} />
      <Star />
      {planets.map((planet, i) => (
        <OrbitPath key={`orbit-${i}`} radius={planet.orbitRadius} />
      ))}
      {planets.map((planet, i) => (
        <Planet
          key={planet.name}
          orbitRadius={planet.orbitRadius}
          size={planet.size}
          color={planet.color}
          orbitSpeed={planet.orbitSpeed}
          isSelected={selectedPlanet === i}
          onClick={() => setSelectedPlanet(i)}
        />
      ))}
      <OrbitControls 
        enablePan={true} 
        enableZoom={true} 
        enableRotate={true} 
        minDistance={3} 
        maxDistance={25}
        autoRotate={false}
        makeDefault
      />
    </>
  );
}

const Visualizer = () => {
  const navigate = useNavigate();
  const [selectedPlanet, setSelectedPlanet] = useState(0);
  const planet = planets[selectedPlanet];

  return (
    <div className="h-screen bg-background flex">
      {/* 3D Canvas */}
      <div className="flex-1 relative">
        {/* Top bar */}
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center gap-4 px-4 py-3 bg-background/80 backdrop-blur-sm border-b border-border">
          <Button variant="ghost" size="icon" onClick={() => navigate("/dashboard")} className="text-muted-foreground hover:text-foreground h-8 w-8">
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="flex items-center gap-2">
            <div onClick={() => navigate("/")} className="flex items-center gap-2 cursor-pointer hover:opacity-80 transition-opacity">
              <img
                src="/cosmik-ai.png"
                alt="cosmik.ai"
                className="h-5 w-5"
              />
              <span className="font-data text-xs font-semibold tracking-wide">cosmik.ai</span>
            </div>
            <span className="font-data text-xs text-muted-foreground">/ VISUALIZER</span>
          </div>
          <div className="flex-1 max-w-sm ml-auto">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground" />
              <Input
                placeholder="Query star system..."
                className="pl-9 bg-card border-border font-data text-xs focus-visible:ring-1 focus-visible:ring-foreground rounded-md h-8"
                defaultValue="Kepler-10"
              />
            </div>
          </div>
        </div>

        {/* WebGL Canvas */}
        <Canvas
          camera={{ position: [0, 8, 15], fov: 50 }}
          gl={{ antialias: true }}
          style={{ background: 'transparent' }}
        >
          <Suspense fallback={null}>
            <Scene selectedPlanet={selectedPlanet} setSelectedPlanet={setSelectedPlanet} />
          </Suspense>
        </Canvas>

        {/* Instructions */}
        <div className="absolute bottom-4 left-4 font-data text-[10px] text-muted-foreground uppercase tracking-wider bg-background/60 backdrop-blur-sm px-3 py-2 rounded-md border border-border">
          Drag to rotate • Scroll to zoom • Click planet to select
        </div>
      </div>

      {/* Right Sidebar */}
      <motion.aside
        initial={{ x: 100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.4 }}
        className="w-[380px] border-l border-border bg-background overflow-y-auto"
      >
        <div className="p-4 space-y-4">
          {/* System info */}
          <div className="panel p-4">
            <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider mb-1">System</p>
            <p className="font-data text-sm font-semibold">Kepler-10</p>
            <p className="text-xs text-muted-foreground mt-1">G-type main-sequence • 173 pc • Draco</p>
          </div>

          {/* Planet selector */}
          <div className="flex gap-2">
            {planets.map((p, i) => (
              <button
                key={p.name}
                onClick={() => setSelectedPlanet(i)}
                className={`flex-1 rounded-md px-3 py-2 font-data text-xs transition-colors ${
                  selectedPlanet === i
                    ? "bg-foreground text-background"
                    : "bg-card border border-border text-muted-foreground hover:text-foreground hover:border-muted-foreground"
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>

          {/* Planet Stats */}
          <div className="panel p-4">
            <div className="flex items-center justify-between mb-4">
              <p className="font-data text-xs uppercase tracking-wider">{planet.name}</p>
              <span className="bg-success/20 text-success px-2 py-0.5 rounded-md font-data text-[10px] uppercase tracking-wider">
                {planet.probability}% PROB
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Period (days)", value: planet.period.toFixed(3) },
                { label: "Size (R⊕)", value: planet.sizeVsEarth.toFixed(2) },
                { label: "a/R★", value: planet.semiMajorAxis.toFixed(4) },
                { label: "T_eq (K)", value: planet.temp.toLocaleString() },
              ].map((s) => (
                <div key={s.label} className="bg-muted p-3 rounded-md">
                  <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider mb-1">{s.label}</p>
                  <p className="font-data text-sm font-semibold">{s.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Light Curve */}
          <div className="panel p-4">
            <p className="font-data text-[10px] text-muted-foreground uppercase tracking-wider mb-4">Light Curve — Transit Signal</p>
            <div className="bg-background rounded-md p-2 border border-border">
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={lightCurveData}>
                  <CartesianGrid strokeDasharray="2 2" stroke="hsl(225 6% 16%)" />
                  <XAxis
                    dataKey="phase"
                    tick={{ fill: "hsl(220 5% 55%)", fontSize: 9, fontFamily: "JetBrains Mono" }}
                    tickLine={{ stroke: "hsl(225 6% 16%)" }}
                    axisLine={{ stroke: "hsl(225 6% 16%)" }}
                    label={{ value: "Phase", position: "bottom", fill: "hsl(220 5% 55%)", fontSize: 9, fontFamily: "JetBrains Mono" }}
                  />
                  <YAxis
                    domain={[0.99, 1.002]}
                    tick={{ fill: "hsl(220 5% 55%)", fontSize: 9, fontFamily: "JetBrains Mono" }}
                    tickLine={{ stroke: "hsl(225 6% 16%)" }}
                    axisLine={{ stroke: "hsl(225 6% 16%)" }}
                    tickFormatter={(v) => v.toFixed(3)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(225 6% 8%)",
                      border: "1px solid hsl(225 6% 16%)",
                      borderRadius: "6px",
                      fontFamily: "JetBrains Mono",
                      fontSize: 10,
                    }}
                    labelStyle={{ color: "hsl(220 5% 55%)" }}
                    itemStyle={{ color: "hsl(220 5% 90%)" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="flux"
                    stroke="hsl(220 5% 70%)"
                    strokeWidth={1.5}
                    dot={false}
                    activeDot={{ r: 3, fill: "hsl(220 5% 90%)" }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="font-data text-[10px] text-muted-foreground mt-2">
              Transit depth: δ = 0.008 | Duration: T₁₄ = 1.56 hr
            </p>
          </div>
        </div>
      </motion.aside>
    </div>
  );
};

export default Visualizer;
