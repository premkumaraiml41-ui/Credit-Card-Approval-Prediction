import sys
import types
import importlib.util
import os
import joblib
import pandas as pd


# Minimal stub for streamlit used during import to avoid UI blocking
def _make_stub_streamlit():
    st = types.ModuleType('streamlit')

    def write(*a, **k):
        return None

    def radio(label, options, index=0, **k):
        return options[0] if options else None

    def slider(label, value=0, min_value=0, max_value=0, step=1, **k):
        return value

    def selectbox(label, options, index=0, **k):
        return options[0] if options else None

    def text_input(label, value=''):
        return value

    def button(label):
        return False

    def markdown(*a, **k):
        return None

    def cache_data(f):
        return f

    st.write = write
    st.radio = radio
    st.slider = slider
    st.selectbox = selectbox
    st.text_input = text_input
    st.button = button
    st.markdown = markdown
    st.cache_data = cache_data
    st.secrets = {}

    # provide components.v1 minimal module to satisfy streamlit_lottie import
    components = types.ModuleType('streamlit.components')
    components.v1 = types.ModuleType('streamlit.components.v1')
    st.components = components
    # register submodules so 'import streamlit.components.v1' works
    sys.modules['streamlit.components'] = components
    sys.modules['streamlit.components.v1'] = components.v1

    return st


def import_app(path):
    # inject stub streamlit module
    sys.modules['streamlit'] = _make_stub_streamlit()
    # inject a minimal streamlit_lottie module to avoid importing the real package
    sl = types.ModuleType('streamlit_lottie')
    def st_lottie_spinner(obj, **kwargs):
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Ctx()

    sl.st_lottie_spinner = st_lottie_spinner
    sys.modules['streamlit_lottie'] = sl

    spec = importlib.util.spec_from_file_location('cc_app', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(repo_root, 'cc_approval_pred.py')
    app = import_app(app_path)

    # full_pipeline is defined in the app
    print('Preparing training data using app pipeline...')
    df_prep = app.full_pipeline(app.train_copy.copy())

    # Ensure features match app prediction input: drop ID and target column if present
    df_prep = df_prep.copy()
    if 'ID' in df_prep.columns:
        df_prep = df_prep.drop(columns=['ID'])
    if 'Is high risk' in df_prep.columns:
        # move target to last column
        y = pd.to_numeric(df_prep['Is high risk'])
        X = df_prep.drop(columns=['Is high risk'])
    else:
        # fallback: assume last column is target
        X = df_prep.iloc[:, :-1]
        y = df_prep.iloc[:, -1]

    print('Training GradientBoostingClassifier on preprocessed data...')
    from sklearn.ensemble import GradientBoostingClassifier

    # sample to speed up training if dataset is large
    max_rows = 20000
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)
        y = y.loc[X.index]

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3)
    clf.fit(X, y)

    os.makedirs(os.path.join(repo_root, 'models'), exist_ok=True)
    model_path = os.path.join(repo_root, 'models', 'gradient_boosting_model.sav')
    joblib.dump(clf, model_path)
    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()
