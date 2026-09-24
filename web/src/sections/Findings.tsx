import { SectionHead } from '../components/ui';

const REPORT = 'https://github.com/KassaSana/CivicQ/blob/master/research/REPORT.md';

/** Headline results from research/REPORT.md (24-setting factorial design, fixed seeds). */
export function Findings() {
  return (
    <section id="findings" className="section">
      <SectionHead title="What the study found" />
      <p className="lede">
        The <a href={REPORT}>research report</a> runs this simulator across 24 office configurations, from 2 to 24
        windows, and asks when the textbook staffing rules go wrong. In short:
      </p>
      <ol className="findings">
        <li>
          <b>The textbook rules are safe but wasteful.</b> None of them understaffed in a statistically significant way.
          SIPP used <span className="num">6.7%</span> more staff-hours than the simulation-based plan on average, and up
          to <span className="num">14.5%</span> more. Staffing each hour for its peak load (OL-max) used{' '}
          <span className="num">18.6%</span> more. Most of the excess goes into the opening hour and the early-afternoon
          ramp, when the real line has not built up yet. (§5.2)
        </li>
        <li>
          <b>Lagging demand fixes about half of it.</b> Shifting the demand curve forward by one mean service time
          recovers roughly half of SIPP’s excess when visits take 8 minutes or more, but little when they take 4. (§5.2)
        </li>
        <li>
          <b>Demand uncertainty costs big offices the most.</b> Allowing for 20% day-to-day uncertainty in demand
          adds <span className="num">8%</span> staff for a 2-window office and <span className="num">22%</span> for a
          24-window office, because a large office’s own random variation is relatively small. (§5.3)
        </li>
        <li>
          <b>“Good service” is a policy choice.</b> The same office with the same demand needs{' '}
          <span className="num">18</span> staff-hours under the loosest reasonable definition of the service target
          and <span className="num">22</span> under the strictest. (§5.4)
        </li>
        <li>
          <b>Shifts matter more than the staffing rule.</b> Real staff work 4- and 8-hour shifts. The usual two-step
          method (set an hourly requirement, then fit shifts to it) costs up to <span className="num">68%</span> more
          paid hours than the ideal hour-by-hour plan. Searching over shift schedules directly with the simulator is
          up to <span className="num">15%</span> cheaper. (§5.5)
        </li>
        <li>
          <b>Appointments help through the shifts.</b> Booking 75% of demand into the quiet hours cuts a large
          office’s roster by <span className="num">24%</span>, but barely changes the hour-by-hour need. A small office
          saves nothing. (§5.7)
        </li>
      </ol>
      <p className="note">
        One methodological note: giving every plan the same simulated visitors (common random numbers) cut the
        variance of plan-against-plan comparisons by a median of 40×. Without it, the searches in the study would
        need about 40 times as many simulated days. (§5.4)
      </p>
    </section>
  );
}
