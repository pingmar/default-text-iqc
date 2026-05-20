from ssm_extract_ufe_viz.dictionary import FeatureDictionary, FeatureRecord


def test_dictionary_json_round_trip(tmp_path):
    path = tmp_path / "dict.json"
    dictionary = FeatureDictionary(
        records=[
            FeatureRecord(
                feature_id=0,
                layer=1,
                vector=[1.0, 0.0],
                top_k_image_indices=[3],
                activation_histogram=([1.0, 2.0], [0.0, 1.0, 2.0]),
            )
        ],
        metadata={"model": "toy"},
    )
    dictionary.save(path)
    loaded = FeatureDictionary.load(path)
    assert loaded.metadata["model"] == "toy"
    assert loaded.records[0].vector == [1.0, 0.0]
    assert loaded.records[0].top_k_image_indices == [3]
