from sklearn.base import BaseEstimator, TransformerMixin


class BinaryMapper(BaseEstimator, TransformerMixin):
	"""Simple transformer that maps 'Yes'/'No' to 1/0.

	This mirrors the `BinaryMapper` used when the pipeline was trained and
	exported. Providing the same class here makes it importable during
	joblib unpickling.
	"""

	def __init__(self, yes_val: int = 1, no_val: int = 0):
		self.yes_val = yes_val
		self.no_val = no_val

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		# copy to avoid mutating input
		X_temp = X.copy()

		def _map_val(v):
			# Normalize string inputs and accept common truthy/falsy variants
			try:
				if isinstance(v, str):
					s = v.strip().lower()
					if s in ("yes", "y", "1", "true", "t"):
						return self.yes_val
					if s in ("no", "n", "0", "false", "f"):
						return self.no_val
					# return original string if not recognized
					return v
				# If already numeric (0/1) or other types, return as-is
				return v
			except Exception:
				return v

		# Apply mapping only to object/string columns for DataFrame input
		if hasattr(X_temp, "select_dtypes"):
			for col in X_temp.select_dtypes(include=["object", "string"]).columns:
				X_temp[col] = X_temp[col].apply(_map_val)
		else:
			# If a Series or similar
			try:
				X_temp = X_temp.apply(_map_val)
			except Exception:
				# Fallback to replace for unexpected types
				X_temp = X_temp.replace({"Yes": self.yes_val, "No": self.no_val})

		return X_temp

 