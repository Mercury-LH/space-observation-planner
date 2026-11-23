import "./RunStationTable.css";

// 站址.txt in data/数据.zip lists longitude, latitude (degrees), and altitude (metres).
const sites = {
  1: [-3.609548, 25.473033, 273],
  2: [30.400058, -2.471157, 1501],
  3: [46.132177, -24.851550, 95],
  4: [75.416003, 57.509922, 136],
  5: [97.555922, 40.170479, 1460],
  6: [109.866507, 26.735799, 600],
  7: [138.420392, -5.096416, 19],
  8: [290.458202, -55.254860, 800],
  9: [249.126539, 52.689702, 768],
  10: [358.445658, 50.951506, 900],
};

function coordinate(value, negative, positive) {
  return `${Math.abs(value).toFixed(2)}° ${value < 0 ? negative : positive}`;
}

export default function RunStationTable({ stations }) {
  return (
    <section className="run-site-list" aria-labelledby="run-site-label">
      <h4 id="run-site-label">
        参与站点 <span>({stations.length})</span>
      </h4>
      <table aria-labelledby="run-site-label">
        <thead>
          <tr>
            {["站点", "经度", "纬度", "海拔"].map(label => (
              <th key={label} scope="col">{label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {stations.map(id => {
            const [longitude, latitude, altitude] = sites[Number(id)];
            const signedLongitude = longitude > 180 ? longitude - 360 : longitude;
            return (
              <tr key={id}>
                <th scope="row">
                  <span className="run-site-list__name">
                    <img src="/assets/planning-station.png" alt="" width="28" height="28" />
                    站点 {String(id).padStart(2, "0")}
                  </span>
                </th>
                <td>{coordinate(signedLongitude, "W", "E")}</td>
                <td>{coordinate(latitude, "S", "N")}</td>
                <td>{altitude} m</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
