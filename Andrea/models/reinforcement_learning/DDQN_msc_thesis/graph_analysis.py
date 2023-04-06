import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# --- matplotlib fixing
mpl.use('TkAgg')


if __name__ == "__main__":

    file_name = "results_.xlsx"
    file_path = os.path.join(os.getcwd(), "results", "2023_02_19__10_52", file_name)
    results = pd.read_excel(file_path, index_col="n_episode")

    ma_window = 150


    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Training Analysis', fontweight='bold', fontsize=20)
    grid = plt.GridSpec(nrows=2, ncols=2, wspace=0.2, hspace=0.3)


    plot0 = fig.add_subplot(grid[0, 0])
    plot0.set_title(f'Agent Learning path | '
                    f'μ = {"{:.2f}%".format(results["agent_nav"].sub(1).mean() * 100)}', fontweight='bold', fontsize=16)
    plot0.plot(results['agent_nav'].sub(1), color='green', alpha=0.5)
    plot0.plot(results['agent_nav'].rolling(ma_window).mean().sub(1), color='orange')
    plot0.grid(linestyle='--', color='silver')
    plot0.set_ylim([-0.7, 0.8])
    plot0.set_ylabel('PNL per episode', fontweight='bold')
    plot0.set_xlabel('Number of Episode', style='italic')
    plot0.legend(['point-in-time', f'MA({ma_window})'])

    plot1 = fig.add_subplot(grid[0, 1])
    plot1.set_title(f'Benchmark (Long-Only) | '
                    f'μ = {"{:.2f}%".format(results["mkt_nav"].sub(1).mean() * 100)}', fontweight='bold', fontsize=16)
    plot1.plot(results['mkt_nav'].sub(1), color='green', alpha=0.5)
    plot1.plot(results['mkt_nav'].rolling(ma_window).mean().sub(1), color='orange')
    plot1.grid(linestyle='--', color='silver')
    plot1.set_ylim([-0.7, 0.8])
    plot1.set_ylabel('PNL per episode', fontweight='bold')
    plot1.set_xlabel('Number of Episode', style='italic')
    plot1.legend(['point-in-time', f'MA({ma_window})'])

    plot3 = fig.add_subplot(grid[1, :])
    plot3.set_title(f"PNL per episode | MA({ma_window})", fontweight='bold', fontsize=16)
    plot3.plot(results[['agent_nav', 'mkt_nav']].rolling(ma_window).mean().sub(1))
    plot3.grid(linestyle='--', color='silver')
    plot3.set_ylabel('PNL per episode', fontweight='bold')
    plot3.set_xlabel('Number of Episode', style='italic')
    plot3.legend(['Agent', 'Benchmark | Long-Only'])

    plt.tight_layout()
    plt.show()
    #
    # results['delta'].rolling(ma_window).mean().plot(color='black', linewidth=2)
    # plt.title('Delta PNL per episode', fontweight='bold', fontsize=20)
    # plt.axhline(0, linestyle='--', color='black')
    # plt.grid(linestyle='--', color='silver')
    # plt.ylabel(f'Agent - Benchmark', fontweight='bold')
    # plt.xlabel('Number of Episode', style='italic')
    # plt.legend([f'delta MA({ma_window})'])
    # plt.tight_layout()
    # plt.show()

    # results['% wins MA(100)'].plot()
    # plt.title('Winning-Rate MA(100)', fontweight='bold', fontsize=20)
    # plt.grid(linestyle='--', color='silver')
    # plt.ylabel('Winning-Rate', style='italic', fontweight='bold')
    # plt.xlabel('Number of Episode', style='italic')
    # plt.tight_layout()
    # plt.show()
























