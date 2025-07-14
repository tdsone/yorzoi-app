import { useEffect, useRef, useState } from "react";
import { Editor, updateEditor, actions } from "@teselagen/ove";
import store from "./store";

import "./App.css";

const makeFlatTrace = (depth) => Float32Array.from({ length: 10 }, () => depth); // 10 Y-samples

function coverageToChrom(sequence, aTraceCoverage = [], tTraceCoverage = []) {
  const len = sequence.length;
  // pad coverage with zeros so it matches seq length
  const aCov = Array.from({ length: len }, (_, i) => aTraceCoverage[i] || 0);
  const tCov = Array.from({ length: len }, (_, i) => tTraceCoverage[i] || 0);

  return {
    baseTraces: Array.from({ length: len }, (_, i) => ({
      aTrace: makeFlatTrace(aCov[i]), // fwd strand
      tTrace: makeFlatTrace(tCov[i]), // rev strand
      gTrace: makeFlatTrace(0), // house keeping gene 1: ACT1
      cTrace: makeFlatTrace(0), // house keeping gene 2: TUB1
    })),
    // qualNums: []   // optional
  };
}

const fetchPredictedCoverage = async (sequence) => {
  console.log("Fetching coverage for", sequence);

  try {
    const response = await fetch(
      "https://tom-ellis-lab--yorzoi-app-fastapi-app.modal.run/generate",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ sequences: [sequence] }),
      }
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Prediction result:", data);

    // Assume single sequence request
    const paddedSeq = Object.keys(data)[0];
    const coverage = data[paddedSeq];

    return {
      paddedSeq,
      coverage,
    };
  } catch (error) {
    console.error("Error making prediction:", error);
    return null;
  }
};

function App() {
  const previousSequenceRef = useRef();
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const seq = "Replace_me_or_upload_your_own_sequence";

    updateEditor(store, "DemoEditor", {
      sequenceData: {
        circular: false,
        sequence: seq,
        features: [{ id: "troelf", name: "Promoter", start: 20, end: 30 }],
        chromatogramData: coverageToChrom(seq),
      },
    });

    const linearOnly = [
      [
        { id: "rail", name: "Linear Map", active: false },
        {
          id: "sequence",
          name: "Synthetic Defined -URA (Illumina)",
          active: true,
        },
      ],
    ];

    store.dispatch(
      actions.panelsShownUpdate(linearOnly, {
        editorName: "DemoEditor",
      })
    );

    store.dispatch(
      actions.updateReadOnlyMode(false, { editorName: "DemoEditor" })
    );

    const handleSequenceChange = () => {
      const state = store.getState();
      const editorState = state.VectorEditor?.DemoEditor;
      if (!editorState) return;

      const { sequence } = editorState.sequenceData;
      const previousSequence = previousSequenceRef.current;

      if (sequence !== previousSequence) {
        console.log("Sequence changed:", sequence);
        previousSequenceRef.current = sequence;

        if (/^[acgtn]+$/i.test(sequence)) {
          setIsLoading(true);
          fetchPredictedCoverage(sequence).then((result) => {
            if (result) {
              const { paddedSeq, coverage } = result;
              const aTraceCoverage = coverage[11];
              const tTraceCoverage = coverage[92];
              const chromatogramData = coverageToChrom(
                paddedSeq,
                aTraceCoverage,
                tTraceCoverage
              );
              updateEditor(store, "DemoEditor", {
                sequenceData: {
                  ...editorState.sequenceData,
                  sequence: paddedSeq,
                  chromatogramData,
                },
              });
            }
            setIsLoading(false);
          });
        }
      }
    };

    const unsubscribe = store.subscribe(handleSequenceChange);

    return () => {
      unsubscribe();
    };
  }, []);

  const editorProps = {
    editorName: "DemoEditor",
    isFullscreen: true,
    showMenuBar: true,
    showReadOnly: true,
  };

  return (
    <div>
      {isLoading && (
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            zIndex: 100,
            background: "rgba(255, 255, 255, 0.8)",
            padding: "20px",
            borderRadius: "10px",
          }}
        >
          Loading...
        </div>
      )}
      <Editor {...editorProps} />
    </div>
  );
}

export default App;
