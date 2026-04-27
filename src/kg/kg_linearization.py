from typing import Dict, List


class PathLinearizer:
    """
    Implements Path-based Linearization.
    Converts 1-hop and 2-hop triples into full path sentences to preserve multi-hop reasoning.
    Format: [Type1] [Name1] [REL] [Type2] [Name2]
    """
    @staticmethod
    def linearize(hop1_triples: List[Dict], hop2_triples: List[Dict]) -> List[str]:
        """
        Create individual path sentences: Anchor -> Hop1 -> Hop2

        Args: 
            hop1_triples: List of dicts representing hop1 edges
            hop2_triples: List of dicts representing hop2 edges

        Returns:
            List of independent string paths
        """
        paths = []

        # Create a mapping from hop1 nodes to its outgoing hop2 triples
        hop2_map = {}
        for t2 in hop2_triples:
            n1 = t2.get("n1")
            if n1 not in hop2_map:
                hop2_map[n1] = []
            hop2_map[n1].append(t2)

        # Tranverse hop1 triples to build the base path
        for t1 in hop1_triples:
            # Fall back to generic 'Entity'
            anchor_type = t1.get("n1_type", "Entity")
            anchor_name = t1.get("n1")
            rel1 = t1.get("r1")
            hop1_type = t1.get("n2_type", "Entity")
            hop1_node = t1.get("n2")

            if not anchor_name or not rel1 or not hop1_node:
                continue

            base_sentence = f"{anchor_type} {anchor_name} {rel1} {hop1_type} {hop1_node}"

            # Check if this hop1 node has any hop2 outgoing triples
            outgoing_hop2 = hop2_map.get(hop1_node, [])

            if not outgoing_hop2:
                paths.append(base_sentence)
            else:
                # Extend into 2-hop paths
                for t2 in outgoing_hop2:
                    rel2 = t2.get("r1")
                    hop2_type = t2.get("n2_type", "Entity")
                    hop2_node = t2.get("n2")
                    if rel2 and hop2_node:
                        # Combine 1-hop and 2-hop path into a complete multi-hop relation
                        full_path = f"{base_sentence} which is {rel2} {hop2_type} {hop2_node}"
                        paths.append(full_path)

        # Remove any exact duplicates
        return list(set(full_path))