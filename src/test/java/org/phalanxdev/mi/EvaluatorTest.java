/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.phalanxdev.mi;

import static junit.framework.TestCase.assertEquals;

import java.io.StringReader;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.phalanxdev.mi.Evaluator.EvalMode;
import org.phalanxdev.mi.utils.DefaultMIMessages;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;

/**
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version 1: $
 */
public class EvaluatorTest {

  protected static final String s_data = "@relation test\n"
      + "@attribute one numeric\n"
      + "@attribute two numeric\n"
      + "@attribute class {red,blue}\n"
      + "@data\n"
      + "1,1,red\n"
      + "2,3,blue\n";

  protected Instances m_data;

  @Before
  public void setup() throws Exception {
    m_data = new Instances(new StringReader(s_data));
  }

  @Test(expected = IllegalStateException.class)
  public void testNoInitialization() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.CROSS_VALIDATION, 1, false, false,
        new DefaultMIMessages());

    evaluator.getEvaluation();
  }

  @Test(expected = IllegalStateException.class)
  public void testIntializationButNoEvaluationPerformed() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, false, false,
        new DefaultMIMessages());

    m_data.setClassIndex(m_data.numAttributes() - 1);
    evaluator.initialize(m_data, new J48());

    evaluator.getEvaluation();
  }

  @Test(expected = Exception.class)
  public void testGetEvalRowMetadataNotInitialized() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, false, false,
        new DefaultMIMessages());

    evaluator.getEvalRowMetadata(null, false);
  }

  @Test
  public void testGetEvalRowMetadataNominalClassBasicMetrics() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, false, false,
        new DefaultMIMessages());

    m_data.setClassIndex(m_data.numAttributes() - 1);
    List<Attribute> metadata = evaluator.getEvalRowMetadata(m_data, false);
    assertEquals(15, metadata.size());
  }

  @Test
  public void testGetEvalRowMetadataNominalClassFullMetrics() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, true, true,
        new DefaultMIMessages());

    m_data.setClassIndex(m_data.numAttributes() - 1);
    List<Attribute> metadata = evaluator.getEvalRowMetadata(m_data, false);

    // adds num class values * num IR and area under curve metrics
    int expectedNumMetrics = 15 + (m_data.classAttribute().numValues() * 8);
    assertEquals(expectedNumMetrics, metadata.size());
  }

  @Test
  public void testGetEvalRowMetadataNumericClassBasicMetrics() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, false, false,
        new DefaultMIMessages());

    m_data.setClassIndex(0);
    List<Attribute> metadata = evaluator.getEvalRowMetadata(m_data, false);
    assertEquals(10, metadata.size());

    // turning on AUC and IR should have no affect
    evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, true, true,
        new DefaultMIMessages());
    metadata = evaluator.getEvalRowMetadata(m_data, false);
    assertEquals(10, metadata.size());
  }

}
