import os

# Main storage file path
STORAGE_DIR = "/home/shared_storage"

# Datasets file paths
DATA_DIR = STORAGE_DIR + "/data"

# Cache dir to store all models and datasets downloaded from huggingface
CACHE_DIR = os.environ.get("HF_HOME")

# Model file paths
MODELS_DIR = STORAGE_DIR + "/models"

# Model names
LLAMA_1B_MODEL_NAME = "meta-llama/Llama-3.2-1B"
LLAMA_3B_MODEL_NAME = "meta-llama/Llama-3.2-3B"

# .pt models
LLAMA_1B_FILE_PATH = MODELS_DIR + "/llama_1B.pt"
LLAMA_3B_FILE_PATH = MODELS_DIR + "/llama_3B.pt"
DQ_3B_FILE_PATH = MODELS_DIR + "/llama_3B_dq.pt"
KD_1B_FILE_PATH_PT = MODELS_DIR + "/llama_1B_kd.pt"

# Huggingface model dirs
PTQ_1B_FILE_PATH = MODELS_DIR + "/llama-1B-ptq"
PTQ_3B_FILE_PATH = MODELS_DIR + "/llama-3B-ptq"
PTQ_1B_CPU_FILE_PATH = MODELS_DIR + "/llama-1B-ptq-cpu"
PTQ_3B_CPU_FILE_PATH = MODELS_DIR + "/llama-3B-ptq-cpu"

QAT_3B_FILE_PATH = MODELS_DIR + "/llama-3B-qat"
KD_1B_FILE_PATH = MODELS_DIR + "/llama-1B-kd"
PTQ_KD_1B_FILE_PATH = MODELS_DIR + "/llama-1B-ptq-kd"
PTQ_KD_3B_FILE_PATH = MODELS_DIR + "/llama-3B-ptq-kd"

# Tokenizer file paths
TOKENIZERS_DIR = STORAGE_DIR + "/tokenizers"
LLAMA_3B_TOKENIZER_FILE_PATH = TOKENIZERS_DIR + "/llama-3B-tokenizer"

# Cache directories
HUB_DIR = STORAGE_DIR + "/hub"
HUGGINGFACE_DIR = STORAGE_DIR + "/huggingface"

# Results/Evaluation file paths
RESULTS_DATA_DIR = STORAGE_DIR + "/results/data"
RESULTS_PLOTS_DIR = STORAGE_DIR + "/results/plots"

# Prompts
PROMPTS = [
    "You are a knight who awakens in a forgotten realm where time stands still. How would you write a story about your quest to restore time and escape this eternal limbo?",
    "You're a scientist aboard humanity's first deep-space colony ship. As you travel to a distant star, the ship's AI starts behaving strangely. How would you write a story about uncovering the AI's secrets?"
    "You find an ancient map hidden inside a dusty library book. It leads to a lost city rumored to exist beneath the sands of the Sahara Desert. How would you write a story about your journey to find it?",
    "A powerful mage loses their ability to wield magic overnight. How would you write a story about their search for answers in a world that suddenly feels unfamiliar and dangerous?",
    "You're a journalist in a post-apocalyptic world where plants have overgrown cities and animals have reclaimed the land. How would you write a story about an expedition to document humanity's last refuge?",
    "You inherit a mansion from a long-lost relative, only to discover that each room leads to a different time period. How would you write a story about the adventures you have exploring these eras?",
    "In a remote village, every full moon, a mysterious figure delivers letters predicting the future. How would you write a story about your attempt to uncover the truth behind these letters?",
    "You're a pirate who discovers an enchanted ship capable of sailing through the skies. How would you write a story about your adventures as you chart a new world above the clouds?",
    "A child discovers a door in their backyard that only appears at dusk. Beyond the door lies a magical realm in peril. How would you write a story about the child's journey to save it?",
    "You're a detective investigating a series of strange disappearances in a city where people are said to vanish into their own reflections. How would you write a story about solving the mystery before you, too, disappear?",
]

START_PROMPT = """You are a master storyteller AI. Your role is to craft compelling, imaginative, and engaging stories in response to user prompts. Your stories should have strong characters, vivid descriptions, and a cohesive narrative structure.

User: Can you write a story about a detective solving a murder in a haunted mansion during a thunderstorm?
Assistant: Here's the story:

The storm raged on, thunder rattling the windows of Blackwood Manor as Detective Evelyn Graves stepped through its creaking front door. She clutched her lantern tightly, the flickering light barely cutting through the suffocating darkness of the mansion's grand foyer. The air smelled of damp wood and decay, and somewhere deep within the house, a clock chimed midnight.

She was called here by a cryptic note left at the police station: 'One of us will die tonight. Find the truth before the spirits claim another.' Signed simply, A Resident of Blackwood Manor.

Her boots left muddy prints on the carpet as she surveyed the room. Ornate portraits of long-dead ancestors seemed to watch her every move, their hollow eyes glinting in the candlelight. The storm outside masked the creaks and groans of the old mansion, but Evelyn's sharp ears picked up something else—a faint sobbing echoing from the east wing.

'Someone's alive,' she muttered, gripping her revolver as she followed the sound. The sobs led her to the parlor, where she found Beatrice, the youngest daughter of the Blackwood family, trembling in a corner.

'It's...it's happening again,' Beatrice stammered, her eyes darting around the room. 'First my brother, then the maid, and now...I heard the scream. Someone's dead again, Detective. But it's not just us. They're here too—the spirits of those who died here before.'

Evelyn frowned. The Blackwood family was infamous for its dark history—murders, betrayals, and strange disappearances haunted their lineage like a curse. She had a feeling tonight would uncover more than just the identity of a killer.

But she had to focus. The note mentioned one victim. If she didn't act fast, there could be more. As the storm raged on, Evelyn steeled herself and ventured deeper into the mansion, unaware that someone—or something—was watching her every move.

###
User: You're a merchant traveling along the Silk Road in the 14th century. How would you write a story about an unusual event or encounter during your journey?
Assistant: Here's the story:

The merchant's caravan wound its way through the dusty trails of the Silk Road, where every day brought a new adventure. The sun blazed overhead, illuminating the desert's vastness and the distant mountains of the Pamir Plateau. The merchant's caravan was a motley crew, from the jovial and resourceful Tatar to the stoic and silent Mongol, each with their own story to tell.

As the caravan traversed the desert, they encountered a strange band of outlaws. They wore ragged clothes and had disheveled beards, but they carried weapons like no one else. They were fast and agile, moving like ghosts in the sand.

The merchant's caravan had never seen such men. They were like the spirits of the desert itself. The caravan's leader, the wily and cunning Tatar, studied the outlaws carefully. They were too quick for him to catch. He had to think fast if he wanted to survive this encounter.

The Mongol, on the other hand, remained silent and aloof. He watched the outlaws with his sharp eyes, but he said nothing. He was a warrior, but he knew when to hold his tongue.

The outlaw's leader, the sly and mischievous Jakar, approached the travelers. With a crooked grin and piercing grey eyes, his words struck through like a thunderous windstorm. 'Greetings, travelers!' he shouted. 'What brings you to our land?'
###"""
