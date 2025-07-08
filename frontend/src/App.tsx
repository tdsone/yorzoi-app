import { Button } from "@/components/ui/button";
import { useState } from "react";
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
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
      <form onSubmit={handleSubmit} className="w-full max-w-2xl space-y-4">
        <Textarea
          placeholder="Enter DNA sequence..."
          value={dnaSequence}
          onChange={(e) => setDnaSequence(e.target.value)}
          className="min-h-[200px]"
        />
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
          <Button type="submit" disabled={isLoading}>
            {isLoading ? "Processing..." : "Predict RNA-seq Coverage"}
          </Button>
        </div>
      </form>
    </div>
  );
}

export default App;
