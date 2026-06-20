def to_binary_map(X_sliced):
    col_map = {
        'Yes': 1,
        'No': 0,
        'No internet service': 0,
        'No phone service': 0
    }
    return X_sliced.apply(lambda x: x.map(col_map).fillna(0))