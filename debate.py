import os
import json
import time
import argparse
from langcodes import Language
from datetime import datetime
from tqdm import tqdm
from groq import Groq
from openai import RateLimitError, APIError, APIConnectionError
import backoff
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

client = Groq()
support_models = ['llama3-8b-8192', 'llama3-70b-8192']
NAME_LIST = ["Affirmative", "Negative", "Moderator"]

class Agent:
    def __init__(self, model_name: str, name: str, temperature: float, sleep_time: float = 0) -> None:
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.memory_lst = []
        self.sleep_time = sleep_time

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError, APIConnectionError), max_tries=20)
    def query(self, messages: list[dict], temperature: float) -> str:
        time.sleep(self.sleep_time)
        assert self.model_name in support_models, f"Not support {self.model_name}. Choices: {support_models}"
        try:
            if self.model_name in support_models:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature
                )
                return response.choices[0].message.content
        except RateLimitError as e:
            if "You exceeded your current quota, please check your plan and billing details" in e.user_message:
                raise OutOfQuotaException()
            elif "Your access was terminated due to violation of our policies" in e.user_message:
                raise AccessTerminatedException()
            else:
                raise e

    def set_meta_prompt(self, meta_prompt: str):
        self.memory_lst.append({"role": "system", "content": meta_prompt})

    def add_event(self, event: str):
        self.memory_lst.append({"role": "user", "content": event})

    def add_memory(self, memory: str):
        self.memory_lst.append({"role": "assistant", "content": memory})
        print(f"----- {self.name} -----\n{memory}\n")

    def ask(self, temperature: float = None):
        return self.query(self.memory_lst, temperature=temperature if temperature else self.temperature)


class DebatePlayer(Agent):
    def __init__(self, model_name: str, name: str, temperature: float, openai_api_key: str, sleep_time: float) -> None:
        super().__init__(model_name, name, temperature, sleep_time)
        self.openai_api_key = openai_api_key


class Debate:
    def __init__(self, model_name: str, temperature: float, num_players: int, save_file_dir: str, openai_api_key: str,
                 prompts_path: str, max_round: int, sleep_time: float) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.num_players = num_players
        self.save_file_dir = save_file_dir
        self.openai_api_key = openai_api_key
        self.max_round = max_round
        self.sleep_time = sleep_time

        self.save_file = self._init_save_file(prompts_path)
        self._init_prompt()

        if not self.save_file['base_translation']:
            self._create_base()

        self.players = self._create_agents()
        self._init_agents()

    def _init_save_file(self, prompts_path):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_file = {
            'start_time': current_time,
            'end_time': '',
            'model_name': self.model_name,
            'temperature': self.temperature,
            'num_players': self.num_players,
            'success': False,
            "src_lng": "Sanskrit",
            "tgt_lng": "Russian",
            'source': '',
            'reference': '',
            'base_translation': '',
            "debate_translation": '',
            "Reason": '',
            "Supported Side": '',
            'players': {},
        }
        prompts = json.load(open(prompts_path))
        save_file.update(prompts)
        return save_file

    def _init_prompt(self):
        def prompt_replace(key):
            self.save_file[key] = self.save_file[key].replace(
                "##src_lng##", self.save_file["src_lng"]
            ).replace(
                "##tgt_lng##", self.save_file["tgt_lng"]
            ).replace(
                "##source##", self.save_file["source"]
            ).replace(
                "##base_translation##", self.save_file["base_translation"]
            )
        prompt_replace("base_prompt")
        prompt_replace("player_meta_prompt")
        prompt_replace("moderator_meta_prompt")
        prompt_replace("judge_prompt_last2")

    def _create_base(self):
        print(f"\n===== Translation Task =====\n{self.save_file['base_prompt']}\n")
        agent = DebatePlayer(
            model_name=self.model_name,
            name='Baseline',
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
            sleep_time=self.sleep_time
        )
        agent.add_event(self.save_file['base_prompt'])
        base_translation = agent.ask()
        agent.add_memory(base_translation)
        self.save_file['base_translation'] = base_translation
        self.save_file['affirmative_prompt'] = self.save_file['affirmative_prompt'].replace(
            "##base_translation##", base_translation
        )
        self.save_file['players'][agent.name] = agent.memory_lst

    def _create_agents(self):
        return [
            DebatePlayer(
                model_name=self.model_name,
                name=name,
                temperature=self.temperature,
                openai_api_key=self.openai_api_key,
                sleep_time=self.sleep_time
            ) for name in NAME_LIST
        ]

    def _init_agents(self):
        self.affirmative = self.players[0]
        self.negative = self.players[1]
        self.moderator = self.players[2]

        self.affirmative.set_meta_prompt(self.save_file['player_meta_prompt'])
        self.negative.set_meta_prompt(self.save_file['player_meta_prompt'])
        self.moderator.set_meta_prompt(self.save_file['moderator_meta_prompt'])

        print(f"===== Debate Round-1 =====\n")
        self.affirmative.add_event(self.save_file['affirmative_prompt'])
        self.aff_ans = self.affirmative.ask()
        self.affirmative.add_memory(self.aff_ans)

        self.negative.add_event(self.save_file['negative_prompt'].replace(
            '##aff_ans##', self.aff_ans
        ))
        self.neg_ans = self.negative.ask()
        self.negative.add_memory(self.neg_ans)

        self.moderator.add_event(self.save_file['moderator_prompt'].replace(
            '##aff_ans##', self.aff_ans
        ).replace(
            '##neg_ans##', self.neg_ans
        ).replace(
            '##round##', 'first'
        ))
        self.mod_ans = self.moderator.ask()
        self.moderator.add_memory(self.mod_ans)
        self.mod_ans = eval(self.mod_ans)

    def _round_dct(self, num: int):
        return {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth', 6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
        }.get(num, "")

    def _save_file_to_json(self, id):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_file_path = os.path.join(self.save_file_dir, f"{id}.json")
        self.save_file['end_time'] = current_time
        json_str = json.dumps(self.save_file, ensure_ascii=False, indent=4)
        with open(save_file_path, 'w') as f:
            f.write(json_str)

    def _broadcast(self, msg: str):
        for player in self.players:
            player.add_event(msg)

    def _speak(self, speaker: str, msg: str):
        if not msg.startswith(f"{speaker}: "):
            msg = f"{speaker}: {msg}"
        for player in self.players:
            if player.name != speaker:
                player.add_event(msg)

    def _ask_and_speak(self, player: DebatePlayer):
        ans = player.ask()
        player.add_memory(ans)
        self._speak(player.name, ans)

    def run(self):
        for round in range(self.max_round - 1):
            if self.mod_ans["debate_translation"] != '':
                break
            else:
                print(f"===== Debate Round-{round + 2} =====\n")
                self.affirmative.add_event(self.save_file['debate_prompt'].replace(
                    '##oppo_ans##', self.neg_ans
                ))
                self.aff_ans = self.affirmative.ask()
                self.affirmative.add_memory(self.aff_ans)

                self.negative.add_event(self.save_file['debate_prompt'].replace(
                    '##oppo_ans##', self.aff_ans
                ))
                self.neg_ans = self.negative.ask()
                self.negative.add_memory(self.neg_ans)

                self.moderator.add_event(self.save_file['moderator_prompt'].replace(
                    '##aff_ans##', self.aff_ans
                ).replace(
                    '##neg_ans##', self.neg_ans
                ).replace(
                    '##round##', self._round_dct(round + 2)
                ))
                self.mod_ans = self.moderator.ask()
                self.moderator.add_memory(self.mod_ans)
                self.mod_ans = eval(self.mod_ans)

        if self.mod_ans["debate_translation"] != '':
            self.save_file.update(self.mod_ans)
            self.save_file['success'] = True
        else:
            self._judge_decision()

        for player in self.players:
            self.save_file['players'][player.name] = player.memory_lst

    def _judge_decision(self):
        judge_player = DebatePlayer(
            model_name=self.model_name,
            name='Judge',
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
            sleep_time=self.sleep_time
        )
        aff_ans = self.affirmative.memory_lst[2]['content']
        neg_ans = self.negative.memory_lst[2]['content']

        judge_player.set_meta_prompt(self.save_file['moderator_meta_prompt'])

        judge_player.add_event(self.save_file['judge_prompt_last1'].replace(
            '##aff_ans##', aff_ans
        ).replace(
            '##neg_ans##', neg_ans
        ))
        ans = judge_player.ask()
        judge_player.add_memory(ans)

        judge_player.add_event(self.save_file['judge_prompt_last2'])
        ans = judge_player.ask()
        judge_player.add_memory(ans)

        ans = eval(ans)
        if ans["debate_translation"] != '':
            self.save_file['success'] = True
        self.save_file.update(ans)
        self.players.append(judge_player)


def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input-file", type=str, required=True, help="Input file path")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="Output file dir")
    parser.add_argument("-lp", "--lang-pair", type=str, required=True, help="Language pair")
    parser.add_argument("-m", "--model-name", type=str, default="llama3-70b-8192", help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=0, help="Sampling temperature")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    current_script_path = os.path.abspath(__file__)
    MAD_path = current_script_path.rsplit("/", 1)[0]

    src_lng, tgt_lng = args.lang_pair.split('-')
    src_full = Language.make(language=src_lng).display_name()
    tgt_full = Language.make(language=tgt_lng).display_name()

    config = json.load(open(f"{MAD_path}/config.json", "r"))

    inputs = open(args.input_file, "r").readlines()
    inputs = [l.strip() for l in inputs]

    save_file_dir = args.output_dir
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)

    for id, input in enumerate(tqdm(inputs)):
        prompts_path = f"{save_file_dir}/{id}-config.json"

        config['source'] = input.split('\t')[0]
        config['reference'] = input.split('\t')[1]
        config['src_lng'] = src_full
        config['tgt_lng'] = tgt_full

        with open(prompts_path, 'w') as file:
            json.dump(config, file, ensure_ascii=False, indent=4)

        debate = Debate(
            model_name=args.model_name,
            temperature=args.temperature,
            num_players=3,
            save_file_dir=save_file_dir,
            openai_api_key=openai_api_key,
            prompts_path=prompts_path,
            max_round=3,
            sleep_time=0
        )
        debate.run()
        debate._save_file_to_json(id)
