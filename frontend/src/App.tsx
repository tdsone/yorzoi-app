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

function App() {
  const [dnaSequence, setDnaSequence] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [organism, setOrganism] = useState("s_cerevisiae");
  const [validationError, setValidationError] = useState<string | null>(null);

  // Validate DNA sequence whenever it changes
  useEffect(() => {
    if (dnaSequence.length > 4992) {
      setValidationError("Sequence must not exceed 4992 characters.");
    } else if (dnaSequence && !/^[acgt]+$/i.test(dnaSequence)) {
      setValidationError(
        "Sequence can only contain characters A, C, G, and T."
      );
    } else {
      setValidationError(null);
    }
  }, [dnaSequence]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Prevent submission if sequence is invalid
    if (validationError || dnaSequence.trim() === "") {
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
          body: JSON.stringify({ sequences: [dnaSequence] }),
        }
      );

      const data = await response.json();
      console.log("Prediction result:", data);
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
          <span
            className={`text-sm ${
              dnaSequence.length > 4992
                ? "text-red-500"
                : "text-muted-foreground"
            }`}
          >
            {dnaSequence.length}/4992
          </span>
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
            disabled={isLoading || !!validationError || dnaSequence === ""}
          >
            {isLoading ? "Processing..." : "Predict RNA-seq Coverage"}
          </Button>
        </div>
      </form>
    </div>
  );
}

export default App;
