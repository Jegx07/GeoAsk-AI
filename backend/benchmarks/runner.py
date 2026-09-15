"""GeoAsk AI — Benchmark Runner.

Runs evaluation datasets against the pipeline and computes metrics.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from app.core.agent import GeoAskAgent
from app.models.schemas import AnalysisRequest, ImageInput
from app.models.enums import InputMode

logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Runs a batch of evaluation samples."""
    
    def __init__(self, results_dir: str = "benchmarks/results"):
        self.agent = GeoAskAgent()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    async def evaluate_vqa(self, dataset: list[dict], name: str = "rsvqa_subset"):
        """Evaluate a VQA dataset.
        
        Format: [{'image_path': str, 'query': str, 'ground_truth': str}]
        """
        logger.info(f"Starting benchmark: {name} ({len(dataset)} samples)")
        results = []
        
        for i, sample in enumerate(dataset):
            logger.info(f"Processing sample {i+1}/{len(dataset)}: {sample['query']}")
            
            try:
                # Simulate input manager processing
                img = await self.agent.input_manager.process_upload(
                    sample['image_path'], 
                    Path(sample['image_path']).name
                )
                
                req = AnalysisRequest(
                    query=sample['query'],
                    images=[img],
                    input_mode=InputMode.SINGLE
                )
                
                res = await self.agent.run(req)
                
                results.append({
                    "sample_id": i,
                    "query": sample['query'],
                    "ground_truth": sample['ground_truth'],
                    "prediction": res.answer,
                    "confidence": res.confidence.overall,
                    "processing_time": res.processing_time_seconds,
                    "status": "success"
                })
            except Exception as exc:
                logger.error(f"Sample {i} failed: {exc}")
                results.append({
                    "sample_id": i,
                    "query": sample['query'],
                    "status": "error",
                    "error": str(exc)
                })
                
        # Save results
        out_path = self.results_dir / f"{name}_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Benchmark complete. Results saved to {out_path}")
        return results

if __name__ == "__main__":
    # Example usage script
    logging.basicConfig(level=logging.INFO)
    print("Benchmark runner available. Create a dataset list to evaluate.")
