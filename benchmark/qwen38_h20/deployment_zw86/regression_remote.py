import importlib.util,json,sys,time,unittest
from pathlib import Path
import sglang
p=Path(PROJECT)
assert Path(sglang.__file__).resolve().is_relative_to(p/'workdir/sglang/python'),sglang.__file__
suite=unittest.TestSuite()
for relative,classes in [('test/registered/unit/utils/test_common.py',['TestLoadImage']),('test/registered/unit/multimodal/test_base_processor_image_decode.py',['TestLoadSingleItemImageDecode'])]:
 spec=importlib.util.spec_from_file_location('localtests_'+str(len(suite._tests)),p/'workdir/sglang'/relative);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 for name in classes:suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(getattr(mod,name)))
start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(suite)
report={'success':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'seconds':time.monotonic()-start,'source':sglang.__file__}
(p/'results/jpeg-regression-tests.json').write_text(json.dumps(report,indent=2)+'\n')
assert result.wasSuccessful() and not result.skipped,report
print(json.dumps(report))
