import { useEffect, useRef, useState } from "react";
import { Editor, updateEditor, actions } from "@teselagen/ove";
import { Spinner } from "@blueprintjs/core";
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
  const [predictionSuccessful, setPredictionSuccessful] = useState(false);

  useEffect(() => {
    const seq = "ACGT: Replace me or upload your own DNA sequence";

    updateEditor(store, "DemoEditor", {
      sequenceData: {
        circular: false,
        sequence: seq,
        features: [{ id: "troelf", name: "Promoter", start: 20, end: 30 }],
        chromatogramData: [],
      },
    });

    const panels = [
      [
        { id: "rail", name: "Linear Map", active: false },
        {
          id: "sequence_0",
          type: "sequence",
          name: "YPD (Nanopore)",
          active: true,
          chromatogramIndex: 0,
        },
        {
          id: "sequence_1",
          type: "sequence",
          name: "Synthetic Defined -URA (Illumina)",
          active: false,
          chromatogramIndex: 1,
        },
        {
          id: "sequence_2",
          type: "sequence",
          name: "BY4741 UBR1 -469A>T (Illumina) SRR21628668",
          active: false,
          chromatogramIndex: 2,
        },
      ],
    ];

    store.dispatch(
      actions.panelsShownUpdate(panels, {
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
          setPredictionSuccessful(false);
          store.dispatch(
            actions.updateReadOnlyMode(true, { editorName: "DemoEditor" })
          );
          fetchPredictedCoverage(sequence).then((result) => {
            if (result) {
              const { paddedSeq, coverage } = result;

              const chromatogramData1 = coverageToChrom(
                paddedSeq,
                coverage[11],
                coverage[92]
              );
              const chromatogramData2 = coverageToChrom(
                paddedSeq,
                coverage[0],
                coverage[81]
              );
              const chromatogramData3 = coverageToChrom(
                paddedSeq,
                coverage[10],
                coverage[91]
              );

              updateEditor(store, "DemoEditor", {
                sequenceData: {
                  ...editorState.sequenceData,
                  sequence: paddedSeq,
                  chromatogramData: [
                    chromatogramData1,
                    chromatogramData2,
                    chromatogramData3,
                  ],
                },
              });
              setPredictionSuccessful(true);
            } else {
              setPredictionSuccessful(false);
            }
            setIsLoading(false);
            store.dispatch(
              actions.updateReadOnlyMode(false, { editorName: "DemoEditor" })
            );
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
    ToolBarProps: {
      toolList: [
        "saveTool",
        //you can override a tool like so:
        "downloadTool",
        "importTool",
        "undoTool",
        "redoTool",
        "cutsiteTool",
        "featureTool",
        // "alignmentTool",
        {
          name: "alignmentTool",
          Dropdown: () => {
            return "Hey!";
          },
        },
        // "oligoTool",
        "orfTool",
        // "viewTool",
        "editTool",
        "findTool",
        "visibilityTool",
        {
          name: "predictTool",
          Icon: isLoading ? (
            <>
              <Spinner size={20} /> Predicting coverage...
            </>
          ) : predictionSuccessful ? (
            <p>Predicted Coverage!</p>
          ) : (
            <p>Predicted coverage</p>
          ),
        },
      ],
    },
  };

  return (
    <div>
      <Editor {...editorProps} />
    </div>
  );
}

export default App;
