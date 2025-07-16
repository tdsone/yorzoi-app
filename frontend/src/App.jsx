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
      gTrace: makeFlatTrace(10), // house keeping gene 1: ACT1
      cTrace: makeFlatTrace(50), // house keeping gene 2: TUB1
    })),
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

const track2idx = {
  "YPD (Nanopore)": 11,
  "Synthetic Defined -URA (Illumina)": 0,
  "BY4741 UBR1 -469A>T (Illumina) SRR21628668": 10,
};

const initialSeq = "ACGT: Replace me or upload your own DNA sequence";

function App() {
  const previousSequenceRef = useRef(initialSeq);
  const [isLoading, setIsLoading] = useState(false);
  const [predictionSuccessful, setPredictionSuccessful] = useState(false);

  const handleSequenceChange = () => {
    const state = store.getState();
    const editorState = state.VectorEditor?.DemoEditor;
    if (!editorState) return;

    console.log("previousSequenceRef.current", previousSequenceRef.current);

    const { sequence } = editorState.sequenceData;
    const previousSequence = previousSequenceRef.current;

    if (sequence !== previousSequence) {
      console.log(
        "Sequence changed!",
        "Previous sequence length",
        previousSequence?.length,
        "New sequence length",
        sequence.length
      );

      previousSequenceRef.current = sequence;

      updateEditor(store, "DemoEditor", {
        sequenceData: {
          ...editorState.sequenceData,
          chromatogramData: [],
        },
      });
      setPredictionSuccessful(false);

      if (/^[acgtn]+$/i.test(sequence)) {
        setIsLoading(true);
        store.dispatch(
          actions.updateReadOnlyMode(true, { editorName: "DemoEditor" })
        );

        fetchPredictedCoverage(sequence).then((result) => {
          if (result) {
            console.log("Fetched result", result);

            const { paddedSeq, coverage } = result;

            const chromatogramData = Object.keys(track2idx).map((track) => {
              const trackIdx = track2idx[track];
              return coverageToChrom(
                paddedSeq,
                coverage[trackIdx],
                coverage[trackIdx + 81]
              );
            });

            console.log(chromatogramData);

            updateEditor(store, "DemoEditor", {
              sequenceData: {
                ...editorState.sequenceData,
                chromatogramData,
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
    } else {
      console.log("handleSequenceChange: sequence is the same, doing nothing");
    }
  };

  useEffect(() => {
    updateEditor(store, "DemoEditor", {
      sequenceData: {
        circular: false,
        sequence: initialSeq,
        features: [{ id: "troelf", name: "Promoter", start: 20, end: 30 }],
        chromatogramData: [],
      },
    });

    const trackPanels = Object.keys(track2idx).map((track, idx) => ({
      id: `sequence_${idx}`,
      type: "sequence",
      name: track,
      active: idx === 0,
      chromatogramIndex: idx,
    }));

    const panels = [
      [{ id: "rail", name: "Linear Map", active: false }, ...trackPanels],
    ];

    store.dispatch(
      actions.panelsShownUpdate(panels, {
        editorName: "DemoEditor",
      })
    );

    store.dispatch(
      actions.updateReadOnlyMode(false, { editorName: "DemoEditor" })
    );

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
        "downloadTool",
        "importTool",
        "undoTool",
        "redoTool",
        "cutsiteTool",
        "featureTool",
        "orfTool",
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
