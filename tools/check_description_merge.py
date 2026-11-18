"""Exercise publication, failure recovery and replay without touching the dataset."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import merge_description_final as merge


class MergeChecks(unittest.TestCase):
    def test_failed_read_then_recovery_and_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            version = data/'review/merged_descriptions'
            version.parent.mkdir()
            final = data/'delivery.jsonl'
            final.write_text('{}\n')
            summary = data/'summary.json'
            summary.write_text('{"complete": false}')
            rows = [{'id':s, 'split':s, 'description_status':'保留原文'} for s in merge.SPLITS]
            with patch.multiple(merge, DATA=data, VERSION=version, FINAL=final), patch.object(merge, 'prepare', return_value=rows):
                with patch.object(merge, 'check_reading', side_effect=ValueError('invalid input')), self.assertRaises(ValueError):
                    merge.merge()
                self.assertFalse(version.exists())
                self.assertEqual(json.loads(summary.read_text()), {'complete':False})
                self.assertFalse(any((data/(s+'.jsonl')).exists() for s in merge.SPLITS))
                with patch.object(merge, 'check_reading') as actual:
                    merge.merge()
                    merge.merge()
                    actual.assert_called_once_with(rows)
                for row in rows:
                    self.assertEqual(json.loads((data/(row['split']+'.jsonl')).read_text()), row)
                active = data/'val.jsonl'
                active.unlink()
                active.write_text('newer version')
                with self.assertRaises(RuntimeError):
                    merge.merge()
                self.assertEqual(active.read_text(), 'newer version')


if __name__ == '__main__':
    unittest.main()
