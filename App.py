"""  File and folder structures
ExplainFutures/
  app.py
  pages/
    1_Upload_and_Data_Health.py
    2_Explore_and_Visualize.py
    3_Time_Models.py
    4_Relations_Pairwise.py
    5_Target_Model.py
    6_Future_Lab.py
    7_Scenarios_NLP.py        # v1 includes this (as you decided)

  core/
    config.py
    constants.py
    utils.py

    io/
      loaders.py             # csv/xlsx/txt ingestion
      validators.py          # time parsing, numeric checks, missingness

    preprocess/
      cleaning.py            # resampling, missing data, transforms
      features.py            # lag features, differencing, scaling

    viz/
      plot_time.py           # single variable plots
      plot_multi_axis.py     # multi-axis comparison plot
      plot_scenario.py       # scenario overlays

    models/
      time_models.py         # time → variable models (interp + ML)
      pair_models.py         # variable ↔ variable
      multivar_models.py     # all → one target
      metrics.py             # RMSE/MAE/MAPE/R2, time-split CV
      uncertainty.py         # intervals, bootstrap, calibration

    engine/
      registry.py            # store/retrieve model artifacts
      propagation.py         # what-if propagation graph logic

    nlp/
      scenario_parse.py      # parse scenario text → structured assumptions
      variable_mapping.py    # synonyms + embedding matching (later)
      scenario_quantify.py   # assumptions → parameter values/ranges

  db/
    supabase_client.py       # connection + helpers
    schema.sql               # optional: create tables
    repo.py                  # CRUD functions: datasets/models/scenarios

  assets/
    style.css                # optional
    icons/

  tests/
    test_loaders.py
    test_time_models.py
    test_propagation.py

  requirements.txt
  README.md

"""
