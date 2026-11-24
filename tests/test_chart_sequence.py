import pandas as pd
import pytest

from src.spc_charts import build_control_chart


def test_rules_receive_subgroups_in_observation_order():
    frame = pd.DataFrame({
        "day": ["1-S1", "2-S1", "10-S1"] * 2,
        "weight": [1, 3, 5, 2, 4, 6],
    })
    chart = build_control_chart(frame, value_col="weight", group_col="day", n=2)
    assert chart.subgroup_labels == ["1-S1", "2-S1", "10-S1"]
    assert chart.xbar.tolist() == [1.5, 3.5, 5.5]


def test_limits_require_the_actual_subgroup_size():
    frame = pd.DataFrame({"day": [1, 1, 2], "weight": [1, 2, 3]})
    with pytest.raises(ValueError, match="2 measurements"):
        build_control_chart(frame, value_col="weight", group_col="day", n=2)
