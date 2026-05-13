from typing import Dict, List
from collections import defaultdict


class PathLinearizer:
    """
    Implements Path-based Linearization.
    Converts 1-hop and 2-hop triples into full path sentences to preserve multi-hop reasoning.
    Format: [Type1] [Name1] [REL] [Type2] [Name2]
    """
    @staticmethod
    def linearize(hop1_triples: List[Dict], hop2_triples: List[Dict]) -> List[Dict]:
        """
        Create individual path sentences: Anchor -> Hop1 -> Hop2

        Args: 
            hop1_triples: List of dicts representing hop1 edges
            hop2_triples: List of dicts representing hop2 edges

        Returns:
            List of dictionaries containing 'text' (the path string) and 'metadata'
        """
        paths = []
        
        # Create mapping from hop-1 target node to the full hop-1 triple, this allows us to join hop-2 back
        # to its anchor node
        hop1_map = defaultdict(list)
        for t in hop1_triples:
            hop1_map[t["n2"]].append(t)

        # Nodes that can be developed into hop-2
        nodes_with_hop2 = {t2["n1"] for t2 in hop2_triples}

        # Linearize 1-hop triples (only keep dead-ends, with no hop-2 extension)
        for t in hop1_triples:
            if t["n2"] not in nodes_with_hop2:
                rel1 = t["r1"].replace("_", " ").lower()
                text = f"[{t.get('n1_type', 'Entity')}] {t['n1']} {rel1} [{t.get('n2_type', 'Entity')}] {t['n2']}."
                paths.append({
                    "text": text,
                    "metadata": {
                        "prefix": text,  # 1-hop has no suffix, so the whole string is the prefix
                        "rel2": None,
                        "target": None
                    }
                })

        # Linearize 2-hop (join hop-1 with hop-2)
        for t2 in hop2_triples:
            middle_node = t2["n1"]
            rel2 = t2["r1"].replace("_", " ").lower()
            target_node = f"[{t2.get('n2_type', 'Entity')}] {t2['n2']}"

            # Find all hop-1 paths that connects to this middle node
            for t1 in hop1_map.get(middle_node, []):
                rel1 = t1["r1"].replace("_", " ").lower()
                prefix = f"[{t1.get('n1_type', 'Entity')}] {t1['n1']} {rel1} [{t1.get('n2_type', 'Entity')}] {t1['n2']}"

                text = f"{prefix} which is {rel2} {target_node}"

                paths.append({
                    "text": text,
                    "metadata": {
                        "prefix": prefix,
                        "rel2": rel2,
                        "target": target_node
                    }
                })
        return paths