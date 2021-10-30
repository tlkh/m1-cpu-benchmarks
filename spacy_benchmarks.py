import time
import tqdm
import spacy

line = """
The Apple M1 is an ARM-based system on a chip (SoC). It was designed by Apple Inc. as a central processing unit (CPU) and graphics processing unit (GPU) for its Macintosh computers and iPad Pro tablets.[3] It also marks the first major change to the instruction set used by Macintosh computers since Apple switched Macs from PowerPC to Intel in 2006. Apple claims that it has the world's fastest CPU core "in low power silicon" and the world's best CPU performance per watt.[3][4]
In addition to Apple's own macOS and iPadOS, initial support for the M1 SoC in the Linux kernel was released on June 27, 2021, with version 5.13.[5]
"""

iterations = 300

nlp = spacy.load("en_core_web_sm")
token_count = len(nlp(line))
char_count = len(line)

model_list = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]
result_list = []

for model in tqdm.tqdm(model_list):
    nlp = spacy.load(model)
    nlp(line)

    time.sleep(2.0)

    st = time.time()

    for i in range(iterations):
        nlp(line)

    et = time.time()

    duration = et-st
    fps = iterations/duration

    result_list.append(fps)

    time.sleep(2.0)

print("")
print("Results")
print("=======")
print("Tokens:", token_count)
print(" Chars:", char_count)
for model, fps in zip(model_list, result_list):
    print(model, "- token/sec:", int(fps*token_count))
