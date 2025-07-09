import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

import CoverageView from "@/components/CoverageView";

function App() {
  const [dnaSequence, setDnaSequence] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [organism, setOrganism] = useState("s_cerevisiae");
  const [validationError, setValidationError] = useState<string | null>(null);
  const [coverage, setCoverage] = useState<number[][] | null>(null);
  const [paddedSeq, setPaddedSeq] = useState<string | null>(null);

  // Helper to strip whitespace (spaces, tabs, newlines)
  const stripWhitespace = (s: string) => s.replace(/\s+/g, "");

  // Validate DNA sequence whenever it changes
  useEffect(() => {
    const cleaned = stripWhitespace(dnaSequence);

    if (cleaned.length > 4992) {
      setValidationError(
        "Sequence must not exceed 4992 characters (excluding whitespace)."
      );
    } else if (cleaned && !/^[acgt]+$/i.test(cleaned)) {
      setValidationError(
        "Sequence can only contain characters A, C, G, and T (ignoring whitespace)."
      );
    } else {
      setValidationError(null);
    }
  }, [dnaSequence]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const cleaned = stripWhitespace(dnaSequence);

    // Prevent submission if sequence is invalid or empty
    if (validationError || cleaned === "") {
      return;
    }
    setIsLoading(true);

    try {
      const response = await fetch(
        "https://tom-ellis-lab--yorzoi-app-fastapi-app.modal.run/generate",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ sequences: [cleaned] }),
        }
      );

      const data = await response.json();
      console.log("Prediction result:", data);

      // For now assume single sequence request
      const padded = Object.keys(data)[0];
      setPaddedSeq(padded);
      setCoverage(data[padded]);
    } catch (error) {
      console.error("Error making prediction:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-svh p-4">
      <h1 className="text-4xl font-bold mb-8">Yorzoi</h1>
      <form onSubmit={handleSubmit} className="w-full max-w-2xl space-y-2">
        <div className="flex justify-end">
          {(() => {
            const cleanedLen = stripWhitespace(dnaSequence).length;
            const delta = 4992 - cleanedLen;
            const leftPad = Math.floor(delta / 2);
            const rightPad = delta - leftPad;

            return (
              <div className="flex items-center gap-2 text-sm">
                <span
                  className={
                    cleanedLen > 4992 ? "text-red-500" : "text-muted-foreground"
                  }
                >
                  {cleanedLen}/4992 bp
                </span>
                {cleanedLen > 0 && cleanedLen < 4992 && (
                  <span className="text-sm text-muted-foreground">
                    {"("}"N" padding{" "}
                    {leftPad === rightPad
                      ? `${leftPad}`
                      : `${leftPad}/${rightPad}`}{" "}
                    bp per side{")"}
                  </span>
                )}
              </div>
            );
          })()}
        </div>

        <Textarea
          placeholder="Enter DNA sequence..."
          value={dnaSequence}
          onChange={(e) => setDnaSequence(e.target.value)}
          className="h-[200px] overflow-y-auto resize-none"
        />

        {validationError && (
          <p className="text-sm text-red-500">{validationError}</p>
        )}

        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <span>Organism:</span>
            <Select value={organism} onValueChange={setOrganism}>
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select organism" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="s_cerevisiae">S. Cerevisiae</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Button
            type="submit"
            disabled={
              isLoading ||
              !!validationError ||
              stripWhitespace(dnaSequence) === ""
            }
          >
            {isLoading ? "Processing..." : "Predict RNA-seq Coverage"}
          </Button>
        </div>
      </form>

      {coverage && paddedSeq && (
        <div className="w-full mt-8">
          <CoverageView sequence={paddedSeq} coverage={coverage} />
        </div>
      )}
    </div>
  );
}

export default App;
