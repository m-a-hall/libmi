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
import static junit.framework.TestCase.assertNotNull;

import java.io.StringReader;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.phalanxdev.mi.Evaluator.EvalMode;
import org.phalanxdev.mi.utils.DefaultLogger;
import org.phalanxdev.mi.utils.DefaultMIMessages;
import org.phalanxdev.mi.utils.DefaultVariables;
import org.phalanxdev.mi.utils.IMILogAdapter;
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
  
  protected static final String s_irisData = "@relation iris\n"
      + "\n"
      + "@attribute sepallength\tnumeric\n"
      + "@attribute sepalwidth \tnumeric\n"
      + "@attribute petallength \tnumeric\n"
      + "@attribute petalwidth\tnumeric\n"
      + "@attribute class \t{Iris-setosa,Iris-versicolor,Iris-virginica}\n"
      + "\n"
      + "@data\n"
      + "5.1,3.5,1.4,0.2,Iris-setosa\n"
      + "4.9,3.0,1.4,0.2,Iris-setosa\n"
      + "4.7,3.2,1.3,0.2,Iris-setosa\n"
      + "4.6,3.1,1.5,0.2,Iris-setosa\n"
      + "5.0,3.6,1.4,0.2,Iris-setosa\n"
      + "5.4,3.9,1.7,0.4,Iris-setosa\n"
      + "4.6,3.4,1.4,0.3,Iris-setosa\n"
      + "5.0,3.4,1.5,0.2,Iris-setosa\n"
      + "4.4,2.9,1.4,0.2,Iris-setosa\n"
      + "4.9,3.1,1.5,0.1,Iris-setosa\n"
      + "5.4,3.7,1.5,0.2,Iris-setosa\n"
      + "4.8,3.4,1.6,0.2,Iris-setosa\n"
      + "4.8,3.0,1.4,0.1,Iris-setosa\n"
      + "4.3,3.0,1.1,0.1,Iris-setosa\n"
      + "5.8,4.0,1.2,0.2,Iris-setosa\n"
      + "5.7,4.4,1.5,0.4,Iris-setosa\n"
      + "5.4,3.9,1.3,0.4,Iris-setosa\n"
      + "5.1,3.5,1.4,0.3,Iris-setosa\n"
      + "5.7,3.8,1.7,0.3,Iris-setosa\n"
      + "5.1,3.8,1.5,0.3,Iris-setosa\n"
      + "5.4,3.4,1.7,0.2,Iris-setosa\n"
      + "5.1,3.7,1.5,0.4,Iris-setosa\n"
      + "4.6,3.6,1.0,0.2,Iris-setosa\n"
      + "5.1,3.3,1.7,0.5,Iris-setosa\n"
      + "4.8,3.4,1.9,0.2,Iris-setosa\n"
      + "5.0,3.0,1.6,0.2,Iris-setosa\n"
      + "5.0,3.4,1.6,0.4,Iris-setosa\n"
      + "5.2,3.5,1.5,0.2,Iris-setosa\n"
      + "5.2,3.4,1.4,0.2,Iris-setosa\n"
      + "4.7,3.2,1.6,0.2,Iris-setosa\n"
      + "4.8,3.1,1.6,0.2,Iris-setosa\n"
      + "5.4,3.4,1.5,0.4,Iris-setosa\n"
      + "5.2,4.1,1.5,0.1,Iris-setosa\n"
      + "5.5,4.2,1.4,0.2,Iris-setosa\n"
      + "4.9,3.1,1.5,0.1,Iris-setosa\n"
      + "5.0,3.2,1.2,0.2,Iris-setosa\n"
      + "5.5,3.5,1.3,0.2,Iris-setosa\n"
      + "4.9,3.1,1.5,0.1,Iris-setosa\n"
      + "4.4,3.0,1.3,0.2,Iris-setosa\n"
      + "5.1,3.4,1.5,0.2,Iris-setosa\n"
      + "5.0,3.5,1.3,0.3,Iris-setosa\n"
      + "4.5,2.3,1.3,0.3,Iris-setosa\n"
      + "4.4,3.2,1.3,0.2,Iris-setosa\n"
      + "5.0,3.5,1.6,0.6,Iris-setosa\n"
      + "5.1,3.8,1.9,0.4,Iris-setosa\n"
      + "4.8,3.0,1.4,0.3,Iris-setosa\n"
      + "5.1,3.8,1.6,0.2,Iris-setosa\n"
      + "4.6,3.2,1.4,0.2,Iris-setosa\n"
      + "5.3,3.7,1.5,0.2,Iris-setosa\n"
      + "5.0,3.3,1.4,0.2,Iris-setosa\n"
      + "7.0,3.2,4.7,1.4,Iris-versicolor\n"
      + "6.4,3.2,4.5,1.5,Iris-versicolor\n"
      + "6.9,3.1,4.9,1.5,Iris-versicolor\n"
      + "5.5,2.3,4.0,1.3,Iris-versicolor\n"
      + "6.5,2.8,4.6,1.5,Iris-versicolor\n"
      + "5.7,2.8,4.5,1.3,Iris-versicolor\n"
      + "6.3,3.3,4.7,1.6,Iris-versicolor\n"
      + "4.9,2.4,3.3,1.0,Iris-versicolor\n"
      + "6.6,2.9,4.6,1.3,Iris-versicolor\n"
      + "5.2,2.7,3.9,1.4,Iris-versicolor\n"
      + "5.0,2.0,3.5,1.0,Iris-versicolor\n"
      + "5.9,3.0,4.2,1.5,Iris-versicolor\n"
      + "6.0,2.2,4.0,1.0,Iris-versicolor\n"
      + "6.1,2.9,4.7,1.4,Iris-versicolor\n"
      + "5.6,2.9,3.6,1.3,Iris-versicolor\n"
      + "6.7,3.1,4.4,1.4,Iris-versicolor\n"
      + "5.6,3.0,4.5,1.5,Iris-versicolor\n"
      + "5.8,2.7,4.1,1.0,Iris-versicolor\n"
      + "6.2,2.2,4.5,1.5,Iris-versicolor\n"
      + "5.6,2.5,3.9,1.1,Iris-versicolor\n"
      + "5.9,3.2,4.8,1.8,Iris-versicolor\n"
      + "6.1,2.8,4.0,1.3,Iris-versicolor\n"
      + "6.3,2.5,4.9,1.5,Iris-versicolor\n"
      + "6.1,2.8,4.7,1.2,Iris-versicolor\n"
      + "6.4,2.9,4.3,1.3,Iris-versicolor\n"
      + "6.6,3.0,4.4,1.4,Iris-versicolor\n"
      + "6.8,2.8,4.8,1.4,Iris-versicolor\n"
      + "6.7,3.0,5.0,1.7,Iris-versicolor\n"
      + "6.0,2.9,4.5,1.5,Iris-versicolor\n"
      + "5.7,2.6,3.5,1.0,Iris-versicolor\n"
      + "5.5,2.4,3.8,1.1,Iris-versicolor\n"
      + "5.5,2.4,3.7,1.0,Iris-versicolor\n"
      + "5.8,2.7,3.9,1.2,Iris-versicolor\n"
      + "6.0,2.7,5.1,1.6,Iris-versicolor\n"
      + "5.4,3.0,4.5,1.5,Iris-versicolor\n"
      + "6.0,3.4,4.5,1.6,Iris-versicolor\n"
      + "6.7,3.1,4.7,1.5,Iris-versicolor\n"
      + "6.3,2.3,4.4,1.3,Iris-versicolor\n"
      + "5.6,3.0,4.1,1.3,Iris-versicolor\n"
      + "5.5,2.5,4.0,1.3,Iris-versicolor\n"
      + "5.5,2.6,4.4,1.2,Iris-versicolor\n"
      + "6.1,3.0,4.6,1.4,Iris-versicolor\n"
      + "5.8,2.6,4.0,1.2,Iris-versicolor\n"
      + "5.0,2.3,3.3,1.0,Iris-versicolor\n"
      + "5.6,2.7,4.2,1.3,Iris-versicolor\n"
      + "5.7,3.0,4.2,1.2,Iris-versicolor\n"
      + "5.7,2.9,4.2,1.3,Iris-versicolor\n"
      + "6.2,2.9,4.3,1.3,Iris-versicolor\n"
      + "5.1,2.5,3.0,1.1,Iris-versicolor\n"
      + "5.7,2.8,4.1,1.3,Iris-versicolor\n"
      + "6.3,3.3,6.0,2.5,Iris-virginica\n"
      + "5.8,2.7,5.1,1.9,Iris-virginica\n"
      + "7.1,3.0,5.9,2.1,Iris-virginica\n"
      + "6.3,2.9,5.6,1.8,Iris-virginica\n"
      + "6.5,3.0,5.8,2.2,Iris-virginica\n"
      + "7.6,3.0,6.6,2.1,Iris-virginica\n"
      + "4.9,2.5,4.5,1.7,Iris-virginica\n"
      + "7.3,2.9,6.3,1.8,Iris-virginica\n"
      + "6.7,2.5,5.8,1.8,Iris-virginica\n"
      + "7.2,3.6,6.1,2.5,Iris-virginica\n"
      + "6.5,3.2,5.1,2.0,Iris-virginica\n"
      + "6.4,2.7,5.3,1.9,Iris-virginica\n"
      + "6.8,3.0,5.5,2.1,Iris-virginica\n"
      + "5.7,2.5,5.0,2.0,Iris-virginica\n"
      + "5.8,2.8,5.1,2.4,Iris-virginica\n"
      + "6.4,3.2,5.3,2.3,Iris-virginica\n"
      + "6.5,3.0,5.5,1.8,Iris-virginica\n"
      + "7.7,3.8,6.7,2.2,Iris-virginica\n"
      + "7.7,2.6,6.9,2.3,Iris-virginica\n"
      + "6.0,2.2,5.0,1.5,Iris-virginica\n"
      + "6.9,3.2,5.7,2.3,Iris-virginica\n"
      + "5.6,2.8,4.9,2.0,Iris-virginica\n"
      + "7.7,2.8,6.7,2.0,Iris-virginica\n"
      + "6.3,2.7,4.9,1.8,Iris-virginica\n"
      + "6.7,3.3,5.7,2.1,Iris-virginica\n"
      + "7.2,3.2,6.0,1.8,Iris-virginica\n"
      + "6.2,2.8,4.8,1.8,Iris-virginica\n"
      + "6.1,3.0,4.9,1.8,Iris-virginica\n"
      + "6.4,2.8,5.6,2.1,Iris-virginica\n"
      + "7.2,3.0,5.8,1.6,Iris-virginica\n"
      + "7.4,2.8,6.1,1.9,Iris-virginica\n"
      + "7.9,3.8,6.4,2.0,Iris-virginica\n"
      + "6.4,2.8,5.6,2.2,Iris-virginica\n"
      + "6.3,2.8,5.1,1.5,Iris-virginica\n"
      + "6.1,2.6,5.6,1.4,Iris-virginica\n"
      + "7.7,3.0,6.1,2.3,Iris-virginica\n"
      + "6.3,3.4,5.6,2.4,Iris-virginica\n"
      + "6.4,3.1,5.5,1.8,Iris-virginica\n"
      + "6.0,3.0,4.8,1.8,Iris-virginica\n"
      + "6.9,3.1,5.4,2.1,Iris-virginica\n"
      + "6.7,3.1,5.6,2.4,Iris-virginica\n"
      + "6.9,3.1,5.1,2.3,Iris-virginica\n"
      + "5.8,2.7,5.1,1.9,Iris-virginica\n"
      + "6.8,3.2,5.9,2.3,Iris-virginica\n"
      + "6.7,3.3,5.7,2.5,Iris-virginica\n"
      + "6.7,3.0,5.2,2.3,Iris-virginica\n"
      + "6.3,2.5,5.0,1.9,Iris-virginica\n"
      + "6.5,3.0,5.2,2.0,Iris-virginica\n"
      + "6.2,3.4,5.4,2.3,Iris-virginica\n"
      + "5.9,3.0,5.1,1.8,Iris-virginica";

  protected Instances m_simpleData;
  protected Instances m_iris;

  @Before
  public void setup() throws Exception {
    m_simpleData = new Instances(new StringReader(s_data));
    m_iris = new Instances(new StringReader(s_irisData));
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

    m_simpleData.setClassIndex(m_simpleData.numAttributes() - 1);
    evaluator.initialize(m_simpleData, new J48());

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

    m_simpleData.setClassIndex(m_simpleData.numAttributes() - 1);
    List<Attribute> metadata = evaluator.getEvalRowMetadata(m_simpleData, false);
    assertEquals(15, metadata.size());
  }

  @Test
  public void testGetEvalRowMetadataNominalClassFullMetrics() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, true, true,
        new DefaultMIMessages());

    m_simpleData.setClassIndex(m_simpleData.numAttributes() - 1);
    List<Attribute> metadata = evaluator.getEvalRowMetadata(m_simpleData, false);

    // adds num class values * num IR and area under curve metrics
    int expectedNumMetrics = 15 + (m_simpleData.classAttribute().numValues() * 8);
    assertEquals(expectedNumMetrics, metadata.size());
  }

  @Test
  public void testGetEvalRowMetadataNumericClassBasicMetrics() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, false, false,
        new DefaultMIMessages());

    m_simpleData.setClassIndex(0);
    List<Attribute> metadata = evaluator.getEvalRowMetadata(m_simpleData, false);
    assertEquals(10, metadata.size());

    // turning on AUC and IR should have no affect
    evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, true, true,
        new DefaultMIMessages());
    metadata = evaluator.getEvalRowMetadata(m_simpleData, false);
    assertEquals(10, metadata.size());
  }

  @Test
  public void testGeneratePercentSplitEvalBasic() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, false, false,
        new DefaultMIMessages());
    m_iris.setClassIndex(m_iris.numAttributes() - 1);
    evaluator.initialize(m_iris, new J48());
    DefaultLogger logger = new DefaultLogger();
    evaluator.performEvaluation(null, logger, new DefaultVariables());

    Object[] evalRow = evaluator.getEvalRow(null, 0, logger);
    assertNotNull(evalRow);
    assertEquals(15, evalRow.length);
    assertNotNull(evaluator.getEvaluation());
  }

  @Test
  public void testGeneratePercentSplitEvalFullMetrics() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, true, true,
        new DefaultMIMessages());
    m_iris.setClassIndex(m_iris.numAttributes() - 1);
    evaluator.initialize(m_iris, new J48());
    DefaultLogger logger = new DefaultLogger();
    evaluator.performEvaluation(null, logger, new DefaultVariables());

    Object[] evalRow = evaluator.getEvalRow(null, 0, logger);
    assertNotNull(evalRow);
    // adds num class values * num IR and area under curve metrics
    int expectedNumMetrics = 15 + (m_iris.classAttribute().numValues() * 8);
    assertEquals(expectedNumMetrics, evalRow.length);
    assertNotNull(evaluator.getEvaluation());
  }
}
