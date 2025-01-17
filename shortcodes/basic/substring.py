class Shortcode():
	"""Slices up the content."""
	def __init__(self,Unprompted):
		self.Unprompted = Unprompted

	def run_block(self, pargs, kwargs, context,content):
		start = int(self.Unprompted.parse_advanced(kwargs["start"],context)) if "start" in kwargs else 0
		end = int(self.Unprompted.parse_advanced(kwargs["end"],context))  if "end" in kwargs else 0
		step = int(self.Unprompted.parse_advanced(kwargs["step"],context))  if "step" in kwargs else 1
		unit = kwargs["unit"] if "unit" in kwargs else "characters"

		if start == -1: start = None
		if end == -1: end = None

		if unit == "words":
			arr = content.split()
			return_string = " ".join(arr[start:end])
		else:
			return_string = content[start:end:step]

		return(return_string)