"""Check and report the next memory-map contract without building its large graph."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Region:
    name: str
    base: int
    size: int

    def __post_init__(self):
        assert self.size > 0 and self.size & (self.size - 1) == 0
        assert self.base % self.size == 0

    @property
    def address_bits(self):
        return self.size.bit_length() - 1

    def prefix(self):
        return [(bit, self.base >> bit & 1) for bit in range(self.address_bits, 32)]


def main():
    regions = [Region('ram', 0x01000000, 8 << 20),
               Region('drom', 0x02000000, 8 << 20),
               Region('rom', 0x00040000, 256 << 10)]
    for i, region in enumerate(regions):
        for other in regions[i + 1:]:
            assert region.base + region.size <= other.base or other.base + other.size <= region.base
    ram, drom, _ = regions
    ram_heads = ram.size // 1024
    drom_heads = drom.size // 8192
    extra_layers = (ram_heads - 32) // 16 + (drom_heads - 16) // 16
    report = {
        'kind': 'Analytical geometry, not a built or executed Doom machine',
        'regions': [{**r.__dict__, 'prefix_literals': r.prefix()} for r in regions],
        'ram_key_heads': ram_heads,
        'drom_key_heads': drom_heads,
        'extrapolated_chip_layers': 11 + extra_layers,
        'logical_value_heads': 191 + 2 * (ram_heads - 32) + 2 * (drom_heads - 16),
        'extrapolated_parameters_same_dense_hf_layout': 2113019200 + extra_layers * 124212928,
        'shareware_wad_bytes': 4196020,
        'zone_bytes': 6 << 20,
        'output_buffer_320x200_bytes': 320 * 200 * 4,
        'unpruned_writable_object_sections_bytes': 297309,
        'reserved_stack_budget_bytes': 64 << 10,
    }
    report['logical_recurrent_fp32_bytes'] = report['logical_value_heads'] * 128 * 128 * 4
    report['allocated_recurrent_fp32_bytes'] = report['extrapolated_chip_layers'] * 32 * 128 * 128 * 4
    report['extrapolated_dense_bf16_parameter_bytes'] = 2 * report['extrapolated_parameters_same_dense_hf_layout']
    report['provisional_ram_budget_bytes'] = sum(report[k] for k in (
        'zone_bytes', 'output_buffer_320x200_bytes', 'unpruned_writable_object_sections_bytes',
        'reserved_stack_budget_bytes'))
    report['provisional_ram_spare_bytes'] = ram.size - report['provisional_ram_budget_bytes']
    report['budget_limit'] = 'Object totals are unpruned; libc, non-zone dynamic allocations, and live stack are not measured.'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'proposed-geometry.json')
    destination = parser.parse_args().out
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
