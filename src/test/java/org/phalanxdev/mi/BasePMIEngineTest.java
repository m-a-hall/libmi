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
import static junit.framework.Assert.assertNotNull;

import org.junit.Test;
import org.phalanxdev.mi.engines.WekaEngine;

public class BasePMIEngineTest {

  PMIEngine m_engine;

  @Test
  public void testInit() {
    PMIEngine.init();

    // In the absence of installed packages, the only Engine available is Weka
    assertEquals(1, PMIEngine.s_availableEngines.size());
  }

  @Test
  public void testGetEngineNames() {
    assertEquals(1, PMIEngine.getEngineNames().size());
    assertEquals("Weka", PMIEngine.getEngineNames().get(0));
  }

  @Test
  public void testInstantiateEngine() throws Exception {
    PMIEngine wekaEngine = PMIEngine.instantiateEngine(WekaEngine.ENGINE_CLASS);
    assertNotNull(wekaEngine);
  }

  @Test
  public void testGetNamedEngineExisting() throws UnsupportedEngineException {
    PMIEngine wekaEng = PMIEngine.getEngine("Weka");
    assertNotNull(wekaEng);
  }

  @Test(expected = UnsupportedEngineException.class)
  public void testGetNamedEngineNonExistent() throws UnsupportedEngineException {
    PMIEngine.getEngine("Goofy");
  }

  @Test(expected = InstantiationException.class)
  public void testInstantiateInvalidEngine() throws Exception {
    PMIEngine.instantiateEngine(String.class.getCanonicalName());
  }
}
