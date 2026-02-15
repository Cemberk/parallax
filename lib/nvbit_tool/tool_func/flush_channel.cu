#include "utils/channel.hpp"

/* Flush device-side channel buffer to host.
 * Launched as a 1-thread kernel at context termination to drain
 * any events still in the channel's device buffer. */
extern "C" __global__ void flush_channel(ChannelDev* ch_dev) {
    ch_dev->flush();
}
