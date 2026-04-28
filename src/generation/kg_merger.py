"""
Uses metadata to cleanly merge KG paths without repeating relationship strings, 
and implements MAX + Bonus scoring for fair context reordering.

"""
from typing import List, Dict


class KGPathMerger:
    @staticmethod
    def merge_top_paths(ranked_results: List[Dict]) -> List[Dict]:
        """
        Takes the final results (unified text and KG retrieval) from the Cross-Encoder.
        Detects KG paths that share the same multi-hop prefix and condenses their suffixes
        into a single, natural-sounding English sentence.
        
        Args:
            ranked_results: List of dictionaries containing 'text', 'source_type', 
                            and 'cross_encoder_score'.
                            
        Returns:
            A cleaned, merged list of dictionaries ready for prompt building.
        """
        merged_results = []
        kg_groups = {}

        for doc in ranked_results:
            if doc["source_type"] != "kg_retrieval":
                merged_results.append(doc)
                continue

            meta = doc["metadata"]

            # 1-hop paths 
            if meta.get("rel2") is None:
                merged_results.append(doc)
                continue

            # 2-hop paths, group by prefix and relation
            key = (meta["prefix"], meta["rel2"])

            if key not in kg_groups:
                kg_groups[key] = {
                    "base_doc": doc,
                    "targets": [meta["target"]],
                    "scores": [doc.get("cross_encoder_score", 0.0)]
                }
            else:
                kg_groups[key]["targets"].append(meta["target"])
                kg_groups[key]["scores"].append(doc.get("cross_encoder_score", 0.0))

        # Reconstruct the merged KG paths
        for (prefix, rel2), group in kg_groups.items():
            targets = group["targets"]
            scores = group["scores"]

            if len(targets) == 1:
                merged_text = f"{prefix} which is {rel2} {targets[0]}."
            else:
                target_str = ", ".join(targets[:-1]) + f", and {targets[-1]}"
                merged_text = f"{prefix} which is {rel2} {target_str}."

            # Density Bonus Scoring
            agg_score = max(scores) + 0.05 * (len(scores) - 1)

            merged_doc = group["base_doc"].copy()
            merged_doc["text"] = merged_text
            merged_doc["cross_encoder_score"] = agg_score
            merged_results.append(merged_doc)

        merged_results.sort(key=lambda x: x.get("cross_encoder_score", 0.0), reverse=True)
        return merged_results