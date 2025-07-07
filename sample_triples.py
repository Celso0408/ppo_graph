import torch

def select_topk_triples(
    model: torch.nn.Module,
    triples: list,
    k: int,
    entity2id: dict,
    predicate2id: dict,
    return_ids: bool = False,
) -> list | tuple[list, list]:
    """Select the top ``k`` scoring triples using ``model``.

    Parameters
    ----------
    model:
        ``TripleScoringModel`` or compatible module that returns a score for
        each triple.
    triples:
        Iterable of dictionaries with ``subject``, ``predicate`` and ``object``
        keys.
    k:
        Number of triples to return.
    entity2id:
        Mapping from entity strings to integer IDs.
    predicate2id:
        Mapping from predicate strings to integer IDs.
    return_ids:
        If ``True``, also return the encoded ID representation of each triple.

    Returns
    -------
    list | tuple[list, list]
        Either the top ``k`` triples or a tuple of triples and their encoded
        IDs if ``return_ids`` is ``True``.
    """

    encoded = []
    valid_triples = []
    for t in triples:
        try:
            s = entity2id[t['subject']]
            p = predicate2id[t['predicate']]
            o = entity2id[t['object']]
            encoded.append([s, p, o])
            valid_triples.append(t)
        except KeyError:
            continue  # skip unknown entities/predicates

    if not encoded:
        return ([], []) if return_ids else []

    triple_tensor = torch.tensor(encoded, dtype=torch.long)
    with torch.no_grad():
        scores = model(triple_tensor)

    topk_indices = torch.topk(scores, k=min(k, len(encoded))).indices.tolist()
    top_triples = [valid_triples[i] for i in topk_indices]
    top_ids = [encoded[i] for i in topk_indices]

    if return_ids:
        return top_triples, top_ids
    else:
        return top_triples

