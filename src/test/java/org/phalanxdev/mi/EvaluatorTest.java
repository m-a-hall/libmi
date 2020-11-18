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

import java.io.StringReader;
import org.junit.Before;
import org.junit.Test;
import org.phalanxdev.mi.Evaluator.EvalMode;
import org.phalanxdev.mi.utils.DefaultMIMessages;
import weka.classifiers.trees.J48;
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
    Evaluator evaluator = new Evaluator(EvalMode.CROSS_VALIDATION, 1, false, false, new DefaultMIMessages());

    evaluator.getEvaluation();
  }

  @Test(expected = IllegalStateException.class)
  public void testIntializationButNoEvaluationPerformed() throws Exception {
    Evaluator evaluator = new Evaluator(EvalMode.PERCENTAGE_SPLIT, 1, false, false, new DefaultMIMessages());

    m_data.setClassIndex(m_data.numAttributes() - 1);
    evaluator.initialize(m_data, new J48());

    evaluator.getEvaluation();
  }
}
