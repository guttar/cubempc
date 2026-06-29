from __future__ import annotations
import json
import pickle
from dataclasses import asdict, dataclass
from typing import Any
PICKLE_PREFIX = b'PKL:'
TRANSPORT_JSON = 'json'
TRANSPORT_OBJECT = 'object'
from cubempc.poly import BiPoly, TriPoly, UniPoly
MAX_LAYER_DELTA = 1
_POLY_TAG = '__poly__'

@dataclass
class Message:
    src_rank: int
    dst_rank: int | None
    src_layer: int
    dst_layer: int
    protocol: str
    instance_id: str
    msg_type: str
    channel: str
    payload: object

    def validate_routing(self) -> None:
        if self.dst_layer - self.src_layer != MAX_LAYER_DELTA:
            raise ValueError(f'only Com_k -> Com_{{k+1}} allowed, got {self.src_layer} -> {self.dst_layer}')
        if self.src_layer < 0 or self.dst_layer < 0:
            raise ValueError('layer indices must be non-negative')

def _poly_to_json(obj: object) -> object:
    if isinstance(obj, UniPoly):
        return {_POLY_TAG: 'UniPoly', 'coeffs': obj.coeffs}
    if isinstance(obj, BiPoly):
        return {_POLY_TAG: 'BiPoly', 'coeffs': obj.coeffs}
    if isinstance(obj, TriPoly):
        return {_POLY_TAG: 'TriPoly', 'coeffs': obj.coeffs}
    if isinstance(obj, dict):
        return {str(k): _poly_to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_poly_to_json(v) for v in obj]
    return obj

def _poly_from_json(obj: object) -> object:
    if isinstance(obj, dict) and _POLY_TAG in obj:
        kind = obj[_POLY_TAG]
        coeffs = obj['coeffs']
        if kind == 'UniPoly':
            return UniPoly(coeffs)
        if kind == 'BiPoly':
            return BiPoly(coeffs)
        if kind == 'TriPoly':
            return TriPoly(coeffs)
        raise ValueError(f'unknown poly type: {kind}')
    if isinstance(obj, dict):
        return {k: _poly_from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_poly_from_json(v) for v in obj]
    return obj

def encode_message(msg: Message, *, transport: str=TRANSPORT_JSON) -> bytes:
    if transport == TRANSPORT_OBJECT:
        return PICKLE_PREFIX + pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    body = asdict(msg)
    body['payload'] = _poly_to_json(msg.payload)
    return json.dumps(body, separators=(',', ':')).encode('utf-8')

def decode_message(data: bytes) -> Message:
    if data.startswith(PICKLE_PREFIX):
        obj = pickle.loads(data[len(PICKLE_PREFIX):])
        if isinstance(obj, Message):
            return obj
        raise TypeError(f'pickle payload is not Message: {type(obj)}')
    obj = json.loads(data.decode('utf-8'))
    return Message(src_rank=int(obj['src_rank']), dst_rank=None if obj.get('dst_rank') is None else int(obj['dst_rank']), src_layer=int(obj['src_layer']), dst_layer=int(obj['dst_layer']), protocol=str(obj['protocol']), instance_id=str(obj['instance_id']), msg_type=str(obj['msg_type']), channel=str(obj['channel']), payload=_poly_from_json(obj['payload']))

def frame_tcp(body: bytes) -> bytes:
    if len(body) > 4294967295:
        raise ValueError(f'message too large for 4-byte length prefix: {len(body)}')
    return len(body).to_bytes(4, 'big') + body

def unframe_tcp_length(data: bytes) -> int:
    if len(data) < 4:
        raise ValueError('need 4 bytes for length prefix')
    return int.from_bytes(data[:4], 'big')

def encode_framed_message(msg: Message) -> bytes:
    return frame_tcp(encode_message(msg))

def decode_framed_message(data: bytes) -> Message:
    return decode_message(data)
LayerMessage = Message

def broadcast_targets(n: int, self_rank: int) -> list[int]:
    return [r for r in range(n) if r != self_rank]