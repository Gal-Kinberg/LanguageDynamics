from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
import torch

@dataclass
class BaseConfig:
    """Base config with serialization helpers."""
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(**data)


@dataclass
class TinyModelConfig(BaseConfig):
    mode: str = None
    n_layers: int = 2
    n_heads: int = 2
    embed_dim: int = 32
    ffn_dim: int = 32
    context_window: int = 32

@dataclass
class TinyTransformerConfig(TinyModelConfig):
    causal_masking: bool = True
    dropout: float = 0.03

@dataclass
class TinyLMConfig(TinyModelConfig):
    mode: str = field(default='LM', init=False)
    vocab: list = field(default_factory=list)
    dropout_self_attention: float = 0.05
    dropout_embed: float = 0.03
    dropout_residual: float = 0.03

@dataclass
class TinyEncoderConfig(TinyLMConfig):
    latent_dim: int = 32
    cls_id: int | None = None

    def __post_init__(self):
        if self.cls_id is None:
            self.cls_id = self.vocab.index('<CLS>')

@dataclass
class TinyDecoderConfig(TinyLMConfig):
    dropout_cross_attention: float = 0.03
    latent_dim: int = 32
    n_latent: int = 4
    sos_id: int | None = None

    def __post_init__(self):
        if self.sos_id is None:
            self.sos_id = self.vocab.index('<SOS>')

@dataclass
class TinyAutoencoderConfig(TinyLMConfig):
    mode: str = field(default='AE', init=False)
    dropout_cross_attention: float = 0.03
    dropout_latent: float = 0.03
    latent_dim: int = 32
    n_latent: int = 4
    sos_id: int | None = None
    cls_id: int | None = None
    encoder_config: TinyEncoderConfig | None = field(default=None, init=False)
    decoder_config: TinyDecoderConfig | None = field(default=None, init=False)

    def __post_init__(self):
        # get <SOS> and <CLS> IDs, if not given
        if self.sos_id is None:
            self.sos_id = self.vocab.index('<SOS>')
        if self.cls_id is None:
            self.cls_id = self.vocab.index('<CLS>')

        # initialize encoder and decoder configs
        self.encoder_config = TinyEncoderConfig(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            embed_dim=self.embed_dim,
            ffn_dim=self.ffn_dim,
            context_window=self.context_window,
            vocab=self.vocab,
            dropout_self_attention=self.dropout_self_attention,
            dropout_embed=self.dropout_embed,
            dropout_residual=self.dropout_residual,
            cls_id=self.cls_id,
            latent_dim=self.latent_dim
        )

        self.decoder_config = TinyDecoderConfig(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            embed_dim=self.embed_dim,
            ffn_dim=self.ffn_dim,
            context_window=self.context_window,
            vocab=self.vocab,
            dropout_self_attention=self.dropout_self_attention,
            dropout_cross_attention=self.dropout_cross_attention,
            dropout_embed=self.dropout_embed,
            dropout_residual=self.dropout_residual,
            sos_id=self.sos_id,
            latent_dim=self.latent_dim,
            n_latent=self.n_latent
        )

@dataclass
class TinyKoopmanAutoencoderConfig(TinyAutoencoderConfig):
    n_diagonals: int = 5

@dataclass
class TinyDVAEConfig(TinyLMConfig):
    mode: str = field(default='DVAE', init=False)
    latent_dim: int = 3
    dropout_latent: float = 0.00
    encoder_config: TinyLMConfig | None = field(default=None, init=False)
    dropout_decoder: float = 0.03
    dropout_transition: float = 0.03
    decoder_ffn_dim: int = 128
    transition_ffn_dim: int = 128
    pooling: str = 'last'  # 'mean' or 'last' or 'rope-mean'
    decoder_ln: bool = False
    transition_ln: bool = False

    def __post_init__(self):

        # initialize encoder and decoder configs
        self.encoder_config = TinyLMConfig(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            embed_dim=self.embed_dim,
            ffn_dim=self.ffn_dim,
            context_window=self.context_window,
            vocab=self.vocab,
            dropout_self_attention=self.dropout_self_attention,
            dropout_embed=self.dropout_embed,
            dropout_residual=self.dropout_residual,
        )

@dataclass
class TrainingConfig(BaseConfig):
    lr: float = 2e-4
    batch_size: int = 128
    grad_clipping: bool = True

@dataclass
class TrainingKoopmanConfig(TrainingConfig):
    reconstruction_coef: float = 1.0
    koopman_coef: float = 1.0
    regularization_coef: float = 0.0001
    teacher_forcing: bool = False

@dataclass
class TrainingDVAEConfig(TrainingConfig):
    reconstruction_coef: float = 1.0
    warmup_steps: int = 10000
    minimal_beta: float = 0.0
    maximal_beta: float = 1.0
    teacher_forcing: bool = True

@dataclass
class DataGenerationConfig(BaseConfig):
    max_depth: int = 4
    min_length: int = 4
    max_length: int = 50

@dataclass
class ExperimentConfig(BaseConfig):
    epochs: int = 500
    checkpoint_path: str | None = None
    save_every: int = 1
    device_index: int = 0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config: TinyModelConfig = field(default_factory=TinyModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    data_generation_config: Optional[DataGenerationConfig] = field(default_factory=DataGenerationConfig)
    model_save_prefix: str | None = None
    LM_checkpoint_path: str | None = None

    def __post_init__(self):
        self.device = f'cuda:{self.device_index}' if torch.cuda.is_available() else 'cpu'


# CONFIG = {
#     'mode': 'KAE',
#     'n_layers': 2,
#     'n_heads': 2,
#     'embed_dim': 32,
#     'ffn_dim': 256,
#     'vocab': ['(', ')', '[', ']', '<BOS>', '<EOS>', '<SOS>', '<CLS>'],
#     'context_window': 32,
#     'latent_dim': 32,
#     'n_latents': 4,
#     'n_diagonals': 5,  # Number of diagonals in the Koopman operator
#     'max_depth': 4,
#     'min_length': 4,
#     'max_length': 50,
#     'batch_size': 64,
#     'epochs': 500,
#     'lr': 1e-4,
#     'reconstruction_coef': 10.0,
#     'koopman_coef': 4.0,
#     'regularization_coef': 0.001,
#     'teacher_forcing': False,
#     'save_every': 1,
#     # 'checkpoint_path': '/content/drive/MyDrive/LanguageDynamics/models/tiny_AE_dyck2_layers_2_embed_32_ffn_dim_256_context_window_32_latent_32_n_latents_4_maxdepth_4_maxlen_50_date_100825_0456_trial_1/ckpt_epoch100.pt',
#     # 'checkpoint_path': 'C:/Users/Gankl/PycharmProjects/LanguageDynamics/models/tiny_AE_dyck2_layers_2_embed_16_ffn_dim_128_context_window_32_latent_32_n_latents_4_maxdepth_4_maxlen_50_date_080825_1626_trial_1/ckpt_epoch109.pt',
#     'checkpoint_path': '/home/galk/LanguageDynamics/models/tiny_KAE_dyck2_layers_2_embed_32_ffn_dim_256_context_window_32_latent_32_n_latents_4_n_diagonals_5_maxdepth_4_maxlen_50_date_190825_1041_trial_1/ckpt_epoch300.pt',
#     # 'checkpoint_path': None,
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu'
# }


# from dataclasses import dataclass, asdict, field
# import json

# @dataclass(frozen=True, kw_only=True)
# class Apparatus:
#     """
#     Represents an apparatus, usually a textual commentary or annotation related to a specific song,
#     line, or passage of text. Ensures immutability with frozen dataclass and keyword-only argument behavior.

#     The ``Apparatus`` class is used to encapsulate information such as the song name, specific line,
#     lemma, source, target, and optional comment. By default, the type is predefined as "apparatus"
#     and cannot be modified externally.

#     :ivar song_name: The name of the song associated with this apparatus.
#     :type song_name: str
#     :ivar line: The line number in the song corresponding to this apparatus.
#     :type line: int
#     :ivar lemma: The lemma or a specific word/phrase in the song being annotated.
#     :type lemma: str
#     :ivar source: The original text or source information for this apparatus.
#     :type source: str
#     :ivar target: The target text or translation/resulting text for this apparatus.
#     :type target: str
#     :ivar comment: An optional comment or annotation related to this apparatus.
#     :type comment: str | None
#     :ivar type: A fixed value indicating the class of the apparatus, defaulting to "apparatus".
#     :type type: str
#     """
#     song_name: str
#     line: int
#     lemma: str
#     source: str
#     target: str
#     comment: str | None = None
#     type: str = field(default="apparatus", init=False)

#     def to_dict(self):
#         """
#         Converts the Apparatus instance to a dictionary.
#         """
#         return asdict(self)

#     def to_json(self):
#         """
#         Converts the Apparatus instance to a JSON string.
#         """
#         return json.dumps(self.to_dict(), indent=4)

# @dataclass(frozen=True)
# class MissingApparatus(Apparatus):
#     type: str = field(default="missing", init=False)

# @dataclass(frozen=True)
# class FullSpellingApparatus(Apparatus):
#     text: str
#     type: str = field(default="full_spelling", init=False)

# @dataclass(frozen=True)
# class LetterSwapApparatus(Apparatus):
#     text: str
#     old_letter: str
#     new_letter: str
#     type: str = field(default="letter_swap", init=False)

# @dataclass(frozen=True)
# class WordSwapApparatus(Apparatus):
#     text: str
#     # new_word: str
#     type: str = field(default="word_swap", init=False)

# @dataclass(frozen=True)
# class OrderSwapApparatus(Apparatus):
#     text: str
#     type: str = field(default="order_swap", init=False)

# @dataclass(frozen=True)
# class DeletionApparatus(Apparatus):
#     deleted: str
#     corrected: str
#     type: str = field(default="deletion", init=False)