# Unprompted by Therefore Games. All Rights Reserved.
# https://patreon.com/thereforegames
# https://github.com/ThereforeGames/unprompted

# This script is intended to be used as an extension for Automatic1111's Stable Diffusion WebUI.

import gradio as gr

import modules.scripts as scripts
from modules.processing import process_images,fix_seed,Processed,process_images,StableDiffusionProcessing
from modules.shared import opts, cmd_opts, state, Options
import modules.shared as shared
from modules.hypernetworks import hypernetwork

import modules.sd_models

import sys
import os
base_dir = scripts.basedir()
sys.path.append(base_dir)
# Main object
from lib.shared import Unprompted
Unprompted = Unprompted(base_dir)

class Scripts(scripts.Script):
	def title(self):
		return "Unprompted"

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def ui(self, is_img2img):
		lbl = gr.HTML(label="umprompted_title",value='<span style="margin-left:13px;" class="text-gray-500 text-[0.855rem] mb-2 block dark:text-gray-200 relative z-40">Unprompted</span>')
		dry_run = gr.Checkbox(label="Dry Run",value=False)
		with gr.Box():
			notice = gr.HTML(label="unprompted_notice",value="<strong>Important:</strong> Secondary shortcode tag syntax has changed from <> to {} in v2.0.0. Please see the <strong><a href='https://github.com/ThereforeGames/unprompted/blob/main/docs/CHANGELOG.md' target=_blank>Changelog</a></strong> for more info.")
		if (os.path.exists(f"{base_dir}/{Unprompted.Config.template_directory}/pro/fantasy_card/main{Unprompted.Config.txt_format}")):
			active_class = ""
			txt = "Show Ad"
		else:
			active_class = "active"
			txt = "Dismiss"
		plug = gr.HTML(label="plug",value=f'<div id="unprompted"><div id="ad" class="{active_class} gr-box border-solid border border-gray-200" style="border-radius:0 0 8px 8px"><a href="https://payhip.com/b/hdgNR" target="_blank"><img src="https://i.ibb.co/1MSpHL4/Fantasy-Card-Template2.png" style="float: left;width: 150px;margin-bottom:10px;"></a><h1 style="font-size: 20px;letter-spacing:0.015em;margin-top:10px;">NEW! <strong>Premium Fantasy Card Template</strong> is now available.</h1><p style="margin:1em 0;">Generate a wide variety of creatures and characters in the style of a fantasy card game. Perfect for heroes, animals, monsters, and even crazy hybrids.</p><a href="https://payhip.com/b/hdgNR" target=_blank><button class="gr-button gr-button-lg gr-button-secondary" title="View premium assets for Unprompted">Learn More ➜</button></a><hr style="margin:1em 0;clear:both;"><p style="max-width:80%"><em>Purchases help fund the continued development of Unprompted. Thank you for your support!</em> ❤</p></div><a id="toggle-ad" href="#" style="float:right;display: inline;position:absolute;right:20px;bottom:20px;">{txt}</a></div>')
		return [lbl, dry_run, notice, plug]

	def process(self, p, lbl, dry_run, notice, plug):
		
		# Reset vars
		original_prompt = p.prompt
		original_negative_prompt = p.negative_prompt
		Unprompted.shortcode_user_vars = {}

		# Extra vars
		Unprompted.shortcode_user_vars["batch_index"] = 0

		# Set up system var support - copy relevant p attributes into shortcode var object
		for att in dir(p):
			if not att.startswith("__"):
				Unprompted.shortcode_user_vars[att] = getattr(p,att)

		Unprompted.shortcode_user_vars["prompt"] = Unprompted.process_string(original_prompt)
		Unprompted.shortcode_user_vars["negative_prompt"] = Unprompted.process_string(Unprompted.shortcode_user_vars["negative_prompt"] if Unprompted.shortcode_user_vars["negative_prompt"] else original_negative_prompt)

		# Apply any updates to system vars
		for att in dir(p):
			if not att.startswith("__") and  att.startswith("negative_prompt"):
				setattr(p,att,Unprompted.shortcode_user_vars[att])
		
		# p.seed never egal to None or -1
		
		#if p.seed is not None and p.seed != -1.0:
		#	for i, val in enumerate(p.all_seeds):
		#		if (Unprompted.is_int(p.seed)): p.seed = int(p.seed)
		#		p.all_seeds[i] = p.seed
		#		print("Seed in process")
		#		print(p.all_seeds[i])
		#		print("-----")
		#else:
		#	p.seed = -1
		#	p.seed = fix_seed(p)

		# Batch support
		if (Unprompted.Config.stable_diffusion.batch_support):
			for i, val in enumerate(p.all_prompts):
				Unprompted.shortcode_user_vars["batch_index"] = i

				if (i == 0): 
					p.all_prompts[0] = Unprompted.shortcode_user_vars["prompt"]
					p.all_negative_prompts[0] = Unprompted.shortcode_user_vars["negative_prompt"]
				else:
					p.all_prompts[i] = Unprompted.process_string(original_prompt)
					p.all_negative_prompts[i] = Unprompted.process_string(Unprompted.shortcode_user_vars["negative_prompt"])
				Unprompted.log(f"Result {i}: {p.all_prompts[i]}",False)
		# Keep the same prompt between runs
		else:
			for i, val in enumerate(p.all_prompts):
				p.all_prompts[i] = Unprompted.shortcode_user_vars["prompt"]
				p.all_negative_prompts[i] = Unprompted.shortcode_user_vars["negative_prompt"]
	
	def process_batch(self, p, lbl, dry_run, notice, plug, **kwargs):
		if (dry_run):
			temp_debug = Unprompted.Config.debug
			Unprompted.Config.debug = True
		
		# Reset var on the first batch
		if kwargs["batch_number"] == 0:
			self.original_prompt = p.prompt
			self.original_negative_prompt = p.negative_prompt
			self.sd_model = p.sd_model

		# Reset vars
		Unprompted.shortcode_user_vars = {}

		# Extra vars
		Unprompted.shortcode_user_vars["batch_index"] = 0

		# Set up system var support - copy relevant p attributes into shortcode var object
		for att in dir(p):
			if not att.startswith("__"):
				Unprompted.shortcode_user_vars[att] = getattr(p,att)
		
		Unprompted.shortcode_user_vars["prompt"] = Unprompted.process_string(self.original_prompt)
		Unprompted.shortcode_user_vars["negative_prompt"] = Unprompted.process_string(Unprompted.shortcode_user_vars["negative_prompt"] if Unprompted.shortcode_user_vars["negative_prompt"] else self.original_negative_prompt)
		
		# Not usefull for now because p.seed never egal to None or -1.0 (seed calculate before enter script I think)
		if p.seed is not None and p.seed != -1.0:
			for i, val in enumerate(kwargs["seeds"]):
				kwargs["seeds"][i] = p.all_seeds[len(kwargs["seeds"])*kwargs["batch_number"]+i]
		else:
			p.seed = -1
			p.seed = fix_seed(p)
		
		# Apply any updates to system vars
		for att in dir(p):
			if not att.startswith("__"):
				if att != "sd_model":
					setattr(p,att,Unprompted.shortcode_user_vars[att])
				# Apply the sd_model like in xy_grid.py
				elif att == "sd_model" and isinstance(Unprompted.shortcode_user_vars[att], str):
					x = Unprompted.shortcode_user_vars["sd_model"]
					info = modules.sd_models.get_closet_checkpoint_match(x)
					if info is None:
						raise RuntimeError(f"Unknown checkpoint: {x}")
					modules.sd_models.reload_model_weights(shared.sd_model, info)
					setattr(p,att,shared.sd_model)
				# Not usefull I think, but maybe
				else:
					setattr(p,att,Unprompted.shortcode_user_vars[att])
			

		# Batch support
		if kwargs["batch_number"] == 0:			
		
			if (Unprompted.Config.stable_diffusion.batch_support):
				for i, val in enumerate(p.all_prompts):
					Unprompted.shortcode_user_vars["batch_index"] = i
					if (i == 0): 
						p.all_prompts[0] = Unprompted.shortcode_user_vars["prompt"]
						p.all_negative_prompts[0] = Unprompted.shortcode_user_vars["negative_prompt"]
					else:
						p.all_prompts[i] = Unprompted.process_string(self.original_prompt)
						p.all_negative_prompts[i] = Unprompted.process_string(Unprompted.shortcode_user_vars["negative_prompt"])
					Unprompted.log(f"Result {i}: {p.all_prompts[i]}",False)
			# Keep the same prompt between runs
			else:
				for i, val in enumerate(p.all_prompts):
					p.all_prompts[i] = Unprompted.shortcode_user_vars["prompt"]
					p.all_negative_prompts[i] = Unprompted.shortcode_user_vars["negative_prompt"]
		
		# Apply the prompt, depending of the batch_number, in the kwargs		
		for i, val in enumerate(kwargs["prompts"]):
			kwargs["prompts"][i] = p.all_prompts[len(kwargs["prompts"])*kwargs["batch_number"]+i]

		# Skips the bulk of inference (note: still produces a blank image)
		if (dry_run):
			p.batch_size = 1
			p.steps = 0
			Unprompted.Config.debug = temp_debug
		# Cleanup routines
		Unprompted.log("Entering Cleanup routine...",False)
		for i in Unprompted.cleanup_routines:
			Unprompted.shortcode_objects[i].cleanup()
		# Extensions do not need to return anything, gg no re
		
	def postprocess(self, p, processed, *args):
		# Reload the user checkpoint
		modules.sd_models.reload_model_weights(self.sd_model)
	
	# After routines
	def run(self, p):
		# After routines
		Unprompted.log("Entering After routine...")
		for i in Unprompted.after_routines:
			Unprompted.shortcode_objects[i].after()
