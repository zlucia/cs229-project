import os
import shutil
import subprocess

# Instructions
# Requires FontForge, download with `brew install fontforge` before running file

# Create fontsvgs directory
fontfiles_path = 'font_files/'
fontsvgs_path = 'font_svgs/'
if not os.path.exists(fontsvgs_path):
	os.makedirs(fontsvgs_path)

# Call executable to convert from font ttfs to font svgs
files_list = [os.fsdecode(f) for f in sorted(os.listdir(fontfiles_path)) if not os.fsdecode(f).startswith('.')]
for filename in files_list:
	font = filename[0:-4]
	callable_stub = 'fontforge -script convert.pe '
	call_string = callable_stub + fontfiles_path + font + '.ttf ' + fontsvgs_path
	subprocess.call(call_string, shell=True)

	generated_path = fontfiles_path + font + '.svg'
	new_path = fontsvgs_path + font + '.svg'
	shutil.move(generated_path, new_path)