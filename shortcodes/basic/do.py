class Shortcode():
	"""It's a do-until loop."""
	def __init__(self,Unprompted):
		self.Unprompted = Unprompted

	def run_block(self, pargs, kwargs, context, content):
		final_string = ""

		while True:
			final_string += self.Unprompted.parse_alt_tags(content,context)
			if (self.Unprompted.parse_advanced(kwargs["until"],context)): return(final_string)