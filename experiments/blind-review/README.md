# blind-review

Bounded independent review and adversarial export controls. Start with REPORT.md.

Historical record. Existing reports and artifacts are immutable and retain their original commit and source bindings. Scripts in this directory capture the original environment; they are not portable release entry points. Use [the release guide](../../docs/REPRODUCE.md) for supported commands.

The absolute symlinks in `controls/` were removed for release. They were pointers into private local exports, not evidence bytes. `../checkpoint-integrity/test_verifier_exports.py` constructs its temporary controls at runtime. Use tier 1 for the story controls. The original default still runs both story and Doom controls when all private inputs are available.
