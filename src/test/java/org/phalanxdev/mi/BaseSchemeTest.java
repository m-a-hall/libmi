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

import static junit.framework.Assert.assertEquals;
import static org.mockito.Mockito.mock;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import weka.core.Instances;

/**
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version 1: $
 */
public class BaseSchemeTest {

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

  Scheme m_baseScheme = new Scheme("Mocked") {
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
    public boolean supportsEnvironmentVariables() {
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

  @BeforeClass
  public static void setUpBeforeClass() throws Exception {
    PMIEngine.init();
  }

  @Before
  public void setup() throws Exception {
    m_dataNoString = new Instances(new StringReader(s_data));
    m_dataWithString = new Instances(new StringReader(s_data2));
  }

  @Test
  public void testAddStringAttributeWarningMessageIfNecessaryNoStringAtt() {
    Scheme baseScheme = mock(Scheme.class);

    List<String> messages = new ArrayList<>();
    baseScheme.addStringAttributeWarningMessageIfNeccessary(m_dataNoString, messages);
    assertEquals(0, messages.size());
  }

  @Test
  public void testAddStringAttributeWarningMessageIfNecessaryStringAttPresent() {
    List<String> messages = new ArrayList<>();

    m_baseScheme.addStringAttributeWarningMessageIfNeccessary(m_dataWithString, messages);
    assertEquals(1, messages.size());
  }
}
