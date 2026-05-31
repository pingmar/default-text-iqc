"""
Semantic labeling of extracted features via targeted probe sentences.

Each feature direction is scored against eight linguistic categories by
projecting per-category mean activations onto the feature vector.
The category whose mean projection has the highest z-score becomes the
feature's semantic_label.

Probe categories are chosen to span the three reference corpora:
  positive_sentiment / negative_sentiment - SST-2 / movie-review domain
  negation                                - cross-domain structural marker
  named_entity_person / named_entity_location - WikiText / NER domain
  passive_voice / comparative / question  - syntactic structure markers
"""
from __future__ import annotations

import numpy as np

from ssm_extract_ufe_text.dictionary import FeatureDictionary, FeatureRecord


# Probe sentence bank - 15 sentences per category, written to span distinct
# regions of Mamba-130M's layer-6 activation space.
PROBE_SENTENCES: dict[str, list[str]] = {
    "positive_sentiment": [
        "This film is absolutely wonderful and deeply moving.",
        "The performances were superb and the story utterly captivating.",
        "A brilliant masterpiece that far exceeds all expectations.",
        "I thoroughly enjoyed every moment of this delightful movie.",
        "The direction is inspired and the script is genuinely flawless.",
        "A heartwarming and uplifting experience from start to finish.",
        "The cast delivers outstanding performances throughout the film.",
        "One of the best films I have seen in many years.",
        "The cinematography is breathtaking and the music sublime.",
        "An extraordinary achievement in storytelling and artistry.",
        "The plot is engaging and the characters are deeply compelling.",
        "A truly remarkable film that resonates long after viewing.",
        "The writing is sharp and consistently entertaining.",
        "Every scene sparkles with genuine wit and honest emotion.",
        "A magnificent triumph that deserves every award it receives.",
    ],
    "negative_sentiment": [
        "This film is dull and painfully boring throughout.",
        "The performances are wooden and thoroughly unconvincing.",
        "A disappointing mess that wastes its otherwise talented cast.",
        "I struggled to stay awake during this tedious production.",
        "The script is lazy and the direction entirely uninspired.",
        "A frustrating and deeply unsatisfying viewing experience.",
        "The plot is incoherent and riddled with glaring holes.",
        "One of the worst films released in recent memory.",
        "The dialogue is stilted and embarrassingly bad.",
        "A catastrophic failure on every conceivable level.",
        "The story is predictable and the twists are frankly absurd.",
        "A thoroughly unpleasant and offensive waste of time.",
        "The pacing is dreadful and the characters are unlikable.",
        "Every scene drags and the ending is deeply unsatisfying.",
        "A terrible film that insults the intelligence of its audience.",
    ],
    "negation": [
        "The film was not as good as the critics had suggested.",
        "She did not enjoy the performance at all.",
        "The story never quite delivers on its initial promise.",
        "He could not understand why the film received such praise.",
        "Nobody in the audience was particularly impressed.",
        "The sequel is nothing like as strong as the original.",
        "There was no clear narrative driving the film forward.",
        "The director has never produced work of this poor quality.",
        "The ending does not resolve the central conflict satisfactorily.",
        "I would not recommend this film to anyone.",
        "The cast failed to bring any depth to their roles.",
        "The plot lacks any coherent structure or purpose.",
        "No amount of technical polish could save this disaster.",
        "The dialogue never rises above the merely functional.",
        "There is hardly anything original in this tired rehash.",
    ],
    "named_entity_person": [
        "Steven Spielberg directed the acclaimed science fiction film.",
        "Meryl Streep delivered a remarkable performance in the lead role.",
        "Barack Obama was elected president of the United States in 2008.",
        "Albert Einstein published his theory of relativity in 1905.",
        "Shakespeare wrote more than thirty plays during his lifetime.",
        "Marie Curie discovered two new elements in her Paris laboratory.",
        "Leonardo da Vinci painted the Mona Lisa in Florence.",
        "Charles Darwin proposed the theory of natural selection.",
        "Winston Churchill led Britain through the Second World War.",
        "Beethoven composed nine symphonies despite his progressive deafness.",
        "Newton described the law of universal gravitation in the Principia.",
        "Picasso co-founded the Cubist movement in modern European art.",
        "Mandela spent twenty-seven years in prison before his release.",
        "Crick and Watson described the double-helix structure of DNA in 1953.",
        "Hemingway wrote The Old Man and the Sea while living in Cuba.",
    ],
    "named_entity_location": [
        "The Amazon rainforest spans much of equatorial South America.",
        "London is the capital and largest city of England.",
        "The Sahara Desert is the largest hot desert on Earth.",
        "Paris is renowned worldwide for its art, culture and cuisine.",
        "The Nile flows northward through Egypt into the Mediterranean Sea.",
        "Mount Everest stands at the border of Nepal and Tibet.",
        "Tokyo is one of the most densely populated cities in the world.",
        "The Pacific Ocean covers more area than all land surfaces combined.",
        "The Alps form a natural boundary between northern Italy and France.",
        "New York City is home to over eight million permanent residents.",
        "The Sydney Opera House stands on the shore of Sydney Harbour.",
        "The Great Wall of China stretches for thousands of kilometres.",
        "The Ganges is a sacred river flowing through northern India.",
        "Antarctica is the coldest and driest continent on Earth.",
        "The Grand Canyon was slowly carved by the Colorado River.",
    ],
    "passive_voice": [
        "The award was presented to the director at the annual ceremony.",
        "The novel was written over a period of ten years.",
        "The policy was implemented by the incoming government last year.",
        "The experiment was conducted under carefully controlled conditions.",
        "The building was designed by a renowned international architect.",
        "The report was submitted to the committee on Friday morning.",
        "The suspect was arrested by officers late on Sunday evening.",
        "The contract was signed by both parties in the boardroom.",
        "The manuscript was discovered in a private archive in Rome.",
        "The bridge was constructed using pre-stressed reinforced concrete.",
        "The final decision was reached after months of careful deliberation.",
        "The painting was painstakingly restored by specialists over several months.",
        "The letter was delivered to the wrong address initially.",
        "The regulation was enforced from the first of January.",
        "The species was formally classified by biologists in the nineteenth century.",
    ],
    "comparative": [
        "This sequel is far better than the original film.",
        "The second half is considerably stronger than the first.",
        "Her performance here is more compelling than anything in the previous entry.",
        "The new version is significantly longer than the theatrical original.",
        "The director's latest work is his most ambitious undertaking to date.",
        "The book is more complex but ultimately far more rewarding.",
        "The remake is inferior in every respect to the revered classic.",
        "Her second novel is darker and more mature than her debut.",
        "The final episode is the weakest in the entire acclaimed series.",
        "This production is far more polished than its immediate predecessor.",
        "The acting is stronger here than in any of the earlier instalments.",
        "The special effects are more convincing than in the first entry.",
        "The script is tighter and considerably more focused than before.",
        "The climax is longer but markedly less effective than expected.",
        "This chapter is the most intellectually demanding of all.",
    ],
    "question": [
        "What is the central theme of this complex and ambitious film?",
        "How did the director manage to convey such raw emotion?",
        "Why did the story take such a completely unexpected turn?",
        "Who was ultimately responsible for the screenplay adaptation?",
        "When was the film first released to international audiences?",
        "Is this really the director's finest achievement to date?",
        "Does the ending justify the slow and deliberate pace of the first act?",
        "Which actor delivers the most memorable performance in the film?",
        "How does the score enhance the emotional impact of key scenes?",
        "What genuinely distinguishes this film from others in the genre?",
        "Can the film succeed without the charisma of its lead actor?",
        "Why has this film received such polarised critical opinion?",
        "Where exactly does the story lose its momentum in the second act?",
        "Has the director noticeably improved since the disappointing previous work?",
        "Are the visual effects sufficient to compensate for the weak narrative?",
    ],
}


def get_probe_categories() -> list[str]:
    return list(PROBE_SENTENCES.keys())


def compute_probe_activations(
    probe,
    layer: int,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """
    Run probe sentences through the model and return per-category activation matrices.

    Returns {category: R^(N_cat, d_model)} - last-token hidden states at `layer`.
    Requires the probe to have hooks registered on the given layer.
    """
    from ssm_extract_ufe_text.corpus import get_tokenizer

    tokenizer = get_tokenizer(probe.config.model_name)
    category_activations: dict[str, np.ndarray] = {}

    for category, sentences in PROBE_SENTENCES.items():
        encoded = tokenizer(
            sentences,
            max_length=probe.config.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        batches = []
        for start in range(0, len(sentences), batch_size):
            batch = {
                "input_ids": encoded["input_ids"][start: start + batch_size],
                "attention_mask": encoded["attention_mask"][start: start + batch_size],
            }
            acts = probe.collect_batch(batch)
            if layer in acts:
                batches.append(acts[layer].numpy())
        if batches:
            category_activations[category] = np.concatenate(batches, axis=0)

    return category_activations


def assign_labels_from_activations(
    category_activations: dict[str, np.ndarray],
    records: list[FeatureRecord],
    min_z_score: float = 0.5,
) -> None:
    """
    Assign semantic_label to each FeatureRecord in-place.

    For each feature direction v_i, the per-category score is the mean projection
    of that category's activations onto v_i.  Scores are z-normalised across
    categories; the winning category must exceed min_z_score, otherwise the
    feature is labelled "mixed" (responds to multiple categories equally).

    Modifies records in-place.
    """
    if not category_activations:
        return

    categories = list(category_activations.keys())
    act_matrices = [category_activations[c] for c in categories]

    for rec in records:
        v = np.array(rec.vector, dtype=float)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            rec.semantic_label = "unknown"
            continue
        v_unit = v / norm_v

        # Mean projection of each category's activations onto the feature direction.
        category_scores = np.array([
            float((acts @ v_unit).mean()) for acts in act_matrices
        ])

        std = category_scores.std()
        if std < 1e-10:
            rec.semantic_label = "mixed"
            continue

        z_scores = (category_scores - category_scores.mean()) / std
        best_idx = int(np.argmax(z_scores))
        rec.semantic_label = categories[best_idx] if z_scores[best_idx] >= min_z_score else "mixed"


def assign_semantic_labels(
    probe,
    fd: FeatureDictionary,
    layer: int,
    batch_size: int = 16,
    min_z_score: float = 0.5,
) -> None:
    """
    End-to-end semantic labeling: run probe sentences -> assign labels.
    Modifies fd in-place.  Requires MambaProbe with hooks on `layer`.
    """
    category_activations = compute_probe_activations(probe, layer, batch_size)
    records = fd.records_for_layer(layer)
    assign_labels_from_activations(category_activations, records, min_z_score)
