import os

from . import DB100K_PATH, DB100K_REASONED_PATH

robot_command = f"""
sudo robot reason --reasoner hermit --input {str(DB100K_PATH / "DB100K.owl")} --axiom-generators "ClassAssertion" --create-new-ontology true --output {str(DB100K_REASONED_PATH / "DB100K_reasoned.owl")}
"""

os.system(robot_command)
