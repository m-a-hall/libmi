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
import static junit.framework.TestCase.assertTrue;

import java.io.StringReader;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.instance.Resample;

/**
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version 1: $
 */
public class BaseSupervisedSchemeTest {

  protected static final String s_data = "@relation test\n"
      + "@attribute one numeric\n"
      + "@attribute two numeric\n"
      + "@data\n";

  protected static final String s_data2 = "@relation test\n"
      + "@attribute one numeric\n"
      + "@attribute two numeric\n"
      + "@attribute three string\n"
      + "@data\n";

  Instances m_dataNoString;
  Instances m_dataWithString;


  SupervisedScheme m_baseScheme;

  @BeforeClass
  public static void setUpBeforeClass() throws Exception {
    PMIEngine.init();
  }

  @Before
  public void setup() throws Exception {
    m_dataNoString = new Instances(new StringReader(s_data));
    m_dataWithString = new Instances(new StringReader(s_data2));

    m_baseScheme  = new SupervisedScheme("Mocked") {
      @Override
      public boolean canHandleData(Instances data, List<String> messages) {
        return false;
      }

      @Override
      public boolean supportsIncrementalTraining() {
        return false;
      }

      @Override
      public boolean supportsResumableTraining() {
        return false;
      }

      @Override
      public boolean canHandleStringAttributes() {
        return false;
      }

      @Override
      public Map<String, Object> getSchemeInfo() throws Exception {
        return null;
      }

      @Override
      public void setSchemeOptions(String[] options) throws Exception {

      }

      @Override
      public String[] getSchemeOptions() {
        return new String[0];
      }

      @Override
      public void setSchemeParameters(Map<String, Map<String, Object>> schemeParameters)
          throws Exception {

      }

      @Override
      public Object getConfiguredScheme(Instances trainingHeader) throws Exception {
        return null;
      }

      @Override
      public void setConfiguredScheme(Object scheme) throws Exception {

      }
    };
  }

  @Test
  public void testSamplingConfigVanillaBaseScheme() throws Exception {
    Map<String, String> samplingConfigs = m_baseScheme.getSamplingConfigs();
    Resample resample = new Resample();
    String className = Resample.class.getCanonicalName();
    String options = Utils.joinOptions(resample.getOptions());
    samplingConfigs.put(className, options);
    Classifier toUse = new J48();

    Classifier result = m_baseScheme.adjustForSamplingAndPreprocessing(m_dataNoString, toUse);

    assertTrue(result instanceof FilteredClassifier);

    Filter embedded = ((FilteredClassifier) result).getFilter();
    assertTrue(embedded instanceof MultiFilter);
    MultiFilter mf = (MultiFilter)embedded;
    assertEquals(1, mf.getFilters().length);
    assertTrue(mf.getFilter(0) instanceof Resample);
  }

  @Test
  public void testSamplingConfigBaseSchemeFilteredClassifierNoMultiFilter() throws Exception {
    Map<String, String> samplingConfigs = m_baseScheme.getSamplingConfigs();
    Resample resample = new Resample();
    String className = Resample.class.getCanonicalName();
    String options = Utils.joinOptions(resample.getOptions());
    samplingConfigs.put(className, options);
    FilteredClassifier toUse = new FilteredClassifier();
    toUse.setClassifier(new J48());
    toUse.setFilter(new Discretize());

    Classifier result = m_baseScheme.adjustForSamplingAndPreprocessing(m_dataNoString, toUse);
    assertTrue(result instanceof FilteredClassifier);

    Filter embedded = ((FilteredClassifier) result).getFilter();
    assertTrue(embedded instanceof MultiFilter);
    MultiFilter mf = (MultiFilter)embedded;

    // MultiFilter should contain the original Discretize and the Resample. These should
    // now be in the order of resample, then discretize
    assertEquals(2, mf.getFilters().length);
    assertTrue(mf.getFilter(0) instanceof Resample);
    assertTrue(mf.getFilter(1) instanceof Discretize);
  }

  public void testSamplingConfigBaseSchemeFilteredClassifierMultiFilter() throws Exception {
    Map<String, String> samplingConfigs = m_baseScheme.getSamplingConfigs();
    Resample resample = new Resample();
    String className = Resample.class.getCanonicalName();
    String options = Utils.joinOptions(resample.getOptions());
    samplingConfigs.put(className, options);
    FilteredClassifier toUse = new FilteredClassifier();
    toUse.setClassifier(new J48());
    MultiFilter mf = new MultiFilter();
    Filter[] f = new Filter[1];
    f[0] = new Discretize();
    mf.setFilters(f);
    toUse.setFilter(mf);

    Classifier result = m_baseScheme.adjustForSamplingAndPreprocessing(m_dataNoString, toUse);
    assertTrue(result instanceof FilteredClassifier);

    Filter embedded = ((FilteredClassifier) result).getFilter();
    assertTrue(embedded instanceof MultiFilter);
    mf = (MultiFilter)embedded;

    // in this case, the original Discretize should have been replaced by the Resample -
    // so a total of one filter in the MultiFilter
    assertEquals(1, mf.getFilters().length);
    assertTrue(mf.getFilter(0) instanceof Resample);
  }

  @Test
  public void setPreprocessConfigVanillaBaseScheme() throws Exception {
    Map<String, String> preprocessConfigs = m_baseScheme.getPreprocessingConfigs();
    Discretize d = new Discretize();
    String className = d.getClass().getCanonicalName();
    String options = Utils.joinOptions(d.getOptions());
    preprocessConfigs.put(className, options);

    Classifier toUse = new J48();

    Classifier result = m_baseScheme.adjustForSamplingAndPreprocessing(m_dataNoString, toUse);

    assertTrue(result instanceof FilteredClassifier);

    Filter embedded = ((FilteredClassifier) result).getFilter();
    assertTrue(embedded instanceof MultiFilter);
    MultiFilter mf = (MultiFilter)embedded;
    assertEquals(1, mf.getFilters().length);
    assertTrue(mf.getFilter(0) instanceof Discretize);
  }

  @Test
  public void setPreprocessiConfigBaseSchemeFilteredClassifierNoMultiFilter() throws Exception {
    Map<String, String> preprocessConfigs = m_baseScheme.getPreprocessingConfigs();
    Discretize d = new Discretize();
    String className = d.getClass().getCanonicalName();
    String options = Utils.joinOptions(d.getOptions());
    preprocessConfigs.put(className, options);

    FilteredClassifier toUse = new FilteredClassifier();
    toUse.setClassifier(new J48());
    toUse.setFilter(new Resample());

    Classifier result = m_baseScheme.adjustForSamplingAndPreprocessing(m_dataNoString, toUse);
    assertTrue(result instanceof FilteredClassifier);

    Filter embedded = ((FilteredClassifier) result).getFilter();
    assertTrue(embedded instanceof MultiFilter);
    MultiFilter mf = (MultiFilter)embedded;

    // MultiFilter should contain the original Resample and then Discretize. These should
    // now be in the order of Discretize and then Resample
    assertEquals(2, mf.getFilters().length);
    assertTrue(mf.getFilter(1) instanceof Resample);
    assertTrue(mf.getFilter(0) instanceof Discretize);
  }

}
