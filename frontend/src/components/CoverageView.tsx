import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Brush,
} from "recharts";

import trackAnnotations from "../../../backend/track_annotation.json";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";

interface CoverageViewProps {
  sequence: string; // padded 4992 bp sequence
  coverage: number[][]; // [tracks][bins] shape (162 x 3000)
}

const CoverageView: React.FC<CoverageViewProps> = ({ sequence, coverage }) => {
  const numTracks = coverage.length;
  const half = Math.floor(numTracks / 2);
  const fiveToThree = coverage.slice(0, half);
  const threeToFive = coverage.slice(half, half * 2);

  // Track names from annotation
  const trackNamesPlus: string[] = (trackAnnotations as any)["+"]; // length >= half

  const [selectedIdx, setSelectedIdx] = React.useState(0);

  // Central 3000-bp window
  const crop = Math.floor((sequence.length - 3000) / 2);
  const centralSeq = sequence.slice(crop, crop + 3000);

  // Helper to generate chart data including base letter
  const asPairChartData = (fwd: number[], rev: number[]) =>
    fwd.map((v, idx) => ({
      bp: centralSeq[idx],
      idx,
      fwd: v,
      rev: rev[idx],
    }));

  return (
    <div className="space-y-6 w-full">
      {/* Track selector */}
      <div className="flex items-center gap-2">
        <span className="text-sm">Track:</span>
        <Select
          value={selectedIdx.toString()}
          onValueChange={(val) => setSelectedIdx(parseInt(val, 10))}
        >
          <SelectTrigger className="w-[250px]">
            <SelectValue placeholder="Select track" />
          </SelectTrigger>
          <SelectContent className="max-h-72 overflow-y-auto">
            {trackNamesPlus.slice(0, half).map((name, idx) => (
              <SelectItem key={idx} value={idx.toString()}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Coverage plots */}
      <div className="space-y-4">
        {(() => {
          const idx = selectedIdx;
          const chartData = asPairChartData(fiveToThree[idx], threeToFive[idx]);
          return (
            <ResponsiveContainer
              key={idx}
              width="100%"
              height={120}
              className="bg-white"
            >
              <LineChart
                data={chartData}
                margin={{ left: 20, right: 20, top: 5, bottom: 5 }}
              >
                <XAxis dataKey="bp" tick={{ fontSize: 10 }} interval={1} />
                <YAxis
                  domain={["auto", "auto"]}
                  tick={{ fontSize: 10 }}
                  width={50}
                />
                <Tooltip
                  contentStyle={{ fontSize: "0.75rem" }}
                  formatter={(value: number, name: string) => [
                    value.toFixed(2),
                    name === "fwd" ? "5'→3'" : "3'→5'",
                  ]}
                  labelFormatter={(label, payload) => {
                    // label is the bp char due to XAxis dataKey
                    const idxVal =
                      payload && payload.length > 0
                        ? payload[0].payload.idx
                        : "";
                    return `Pos ${idxVal} : ${label}`;
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="fwd"
                  stroke="#1e40af" /* blue-800 */
                  strokeWidth={1}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="rev"
                  stroke="#047857" /* emerald-700 */
                  strokeWidth={1}
                  dot={false}
                />

                {/* Slider for zoom */}
                <Brush
                  dataKey="idx"
                  height={20}
                  travellerWidth={8}
                  stroke="#8884d8"
                  startIndex={0}
                  endIndex={2999}
                />
              </LineChart>
            </ResponsiveContainer>
          );
        })()}
      </div>
    </div>
  );
};

export default CoverageView;
